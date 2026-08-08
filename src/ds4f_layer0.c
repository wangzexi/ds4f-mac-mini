#include "ds4f_gguf.h"
#include "ds4f_quant.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const ds4f_tensor *need(const ds4f_gguf *g, const char *name) {
    const ds4f_tensor *t = ds4f_gguf_find(g, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    return t;
}
static void load(const ds4f_gguf *g, const ds4f_tensor *t, void **p) {
    if (ds4f_tensor_load(g, t, p)) { perror(t->name); exit(1); }
}
static void stats(const char *name, const float *x, size_t n) {
    float lo=x[0], hi=x[0], ss=0.0f;
    for(size_t i=0;i<n;++i){if(x[i]<lo)lo=x[i];if(x[i]>hi)hi=x[i];ss+=x[i]*x[i];}
    printf("%s: min=%.6g max=%.6g rms=%.6g\n",name,lo,hi,sqrtf(ss/(float)n));
}
static void hc_split(float *out,const float *mix,const float *scale,const float *base) {
    const int n=4; const float eps=1e-6f;
    for(int i=0;i<n;++i) out[i]=1.0f/(1.0f+expf(-(mix[i]*scale[0]+base[i])))+eps;
    for(int i=0;i<n;++i) out[n+i]=2.0f/(1.0f+expf(-(mix[n+i]*scale[1]+base[n+i])));
    float c[16];
    for(int dst=0;dst<n;++dst){float mx=-INFINITY;
        for(int src=0;src<n;++src){int k=src+dst*n;float v=mix[2*n+k]*scale[2]+base[2*n+k];c[k]=v;if(v>mx)mx=v;}
        float sum=0;for(int src=0;src<n;++src){int k=src+dst*n;c[k]=expf(c[k]-mx);sum+=c[k];}
        for(int src=0;src<n;++src){int k=src+dst*n;c[k]=c[k]/sum+eps;}
    }
    for(int src=0;src<n;++src){float sum=0;for(int dst=0;dst<n;++dst)sum+=c[src+dst*n];for(int dst=0;dst<n;++dst)c[src+dst*n]/=sum+eps;}
    for(int it=1;it<20;++it){
        for(int dst=0;dst<n;++dst){float sum=0;for(int src=0;src<n;++src)sum+=c[src+dst*n];for(int src=0;src<n;++src)c[src+dst*n]/=sum+eps;}
        for(int src=0;src<n;++src){float sum=0;for(int dst=0;dst<n;++dst)sum+=c[src+dst*n];for(int dst=0;dst<n;++dst)c[src+dst*n]/=sum+eps;}
    }
    memcpy(out+8,c,sizeof(c));
}
static void weighted(float *out,const float *x,const float *w) {
    for(size_t d=0;d<4096;++d){out[d]=0;for(int h=0;h<4;++h)out[d]+=x[(size_t)h*4096+d]*w[h];}
}
static void hc_post(float *out,const float *block,const float *res,const float *post,const float *comb) {
    for(int dst=0;dst<4;++dst)for(size_t d=0;d<4096;++d){float v=block[d]*post[dst];for(int src=0;src<4;++src)v+=comb[dst+src*4]*res[(size_t)src*4096+d];out[(size_t)dst*4096+d]=v;}
}
static float softplus(float x) { if(x>20.0f)return x; if(x<-20.0f)return expf(x); return log1pf(expf(x)); }
static void select_top6(const float *score,const float *bias,int *sel) {
    for(int i=0;i<6;++i)sel[i]=-1;
    for(int i=0;i<256;++i)for(int j=0;j<6;++j)if(sel[j]<0||score[i]+(bias?bias[i]:0)>score[sel[j]]+(bias?bias[sel[j]]:0)){
        for(int k=5;k>j;--k)sel[k]=sel[k-1];sel[j]=i;break;
    }
}
static void moe_layer0(const ds4f_gguf *g,const float *inp,int token,float *out) {
    const size_t E=4096,M=2048; float *ffncur=malloc(E*4),*norm=malloc(E*4),*flat=malloc(E*4*4),*mix=malloc(24*4),*split=malloc(24*4),*moe=calloc(E,4),*shared=malloc(E*4),*gate=malloc(M*4),*up=malloc(M*4),*mid=malloc(M*4),*down=malloc(E*4);
    float *res=malloc(E*4*4);memcpy(res,inp,E*4*4);ds4f_rms_norm(flat,res,NULL,E*4,1e-6f);
    ds4f_matvec(g,need(g,"blk.0.hc_ffn_fn.weight"),flat,mix);float *scale=NULL,*base=NULL,*nw=NULL;load(g,need(g,"blk.0.hc_ffn_scale.weight"),(void**)&scale);load(g,need(g,"blk.0.hc_ffn_base.weight"),(void**)&base);hc_split(split,mix,scale,base);weighted(ffncur,res,split);load(g,need(g,"blk.0.ffn_norm.weight"),(void**)&nw);ds4f_rms_norm(norm,ffncur,nw,E,1e-6f);stats("ffn_cur",ffncur,E);stats("ffn_norm",norm,E);
    float *logits=malloc(256*4);ds4f_matvec(g,need(g,"blk.0.ffn_gate_inp.weight"),norm,logits);float probs[256];for(int i=0;i<256;++i)probs[i]=sqrtf(softplus(logits[i]));int sel[6];int32_t hash[6];const ds4f_tensor *ht=ds4f_gguf_find(g,"blk.0.ffn_gate_tid2eid.weight");if(ht){if(ds4f_gguf_read(g,ht,(uint64_t)token*6*4,hash,24)){perror("hash route");exit(1);}for(int i=0;i<6;++i)sel[i]=hash[i];}else{float *bias=NULL;const ds4f_tensor *bt=ds4f_gguf_find(g,"blk.0.exp_probs_b.bias");if(bt)load(g,bt,(void**)&bias);select_top6(probs,bias,sel);free(bias);}float sum=0;float ew[6];for(int i=0;i<6;++i){ew[i]=probs[sel[i]];sum+=ew[i];}for(int i=0;i<6;++i)ew[i]=ew[i]/fmaxf(sum,6.103515625e-5f)*1.5f;
    const ds4f_tensor *gt=need(g,"blk.0.ffn_gate_exps.weight"),*ut=need(g,"blk.0.ffn_up_exps.weight"),*dt=need(g,"blk.0.ffn_down_exps.weight");for(int s=0;s<6;++s){ds4f_matvec_expert_q8k(g,gt,(uint32_t)sel[s],norm,E,gate);ds4f_matvec_expert_q8k(g,ut,(uint32_t)sel[s],norm,E,up);ds4f_swiglu(mid,gate,up,M,10.0f);for(size_t i=0;i<M;++i)mid[i]*=ew[s];ds4f_matvec_expert_q8k(g,dt,(uint32_t)sel[s],mid,M,down);for(size_t i=0;i<E;++i)moe[i]+=down[i];}
    ds4f_matvec(g,need(g,"blk.0.ffn_gate_shexp.weight"),norm,gate);ds4f_matvec(g,need(g,"blk.0.ffn_up_shexp.weight"),norm,up);ds4f_swiglu(mid,gate,up,M,10.0f);ds4f_matvec(g,need(g,"blk.0.ffn_down_shexp.weight"),mid,shared);stats("routed_moe",moe,E);stats("shared_ffn",shared,E);for(size_t i=0;i<E;++i)shared[i]+=moe[i];hc_post(out,shared,res,split+4,split+8);stats("after_ffn_hc",out,E*4);
    free(down);free(mid);free(up);free(gate);free(shared);free(moe);free(split);free(mix);free(flat);free(res);free(nw);free(base);free(scale);free(ffncur);free(norm);free(logits);
}

int main(int argc,char **argv){
    if(argc<2){fprintf(stderr,"usage: %s MODEL.gguf [token]\n",argv[0]);return 2;}
    int token=argc>2?atoi(argv[2]):67;
    ds4f_gguf g; if(ds4f_gguf_open(&g,argv[1])){perror("GGUF");return 1;}
    const size_t E=4096, HC=4, HD=512, Q=64*512, QR=1024;
    uint16_t *emb16=malloc(E*2); float *emb=malloc(E*4), *res=malloc(E*HC*4), *flat=malloc(E*HC*4), *mix=malloc(24*4), *split=malloc(24*4);
    const ds4f_tensor *te=need(&g,"token_embd.weight");
    if(ds4f_gguf_read(&g,te,(uint64_t)token*E*2,emb16,E*2)){perror("embedding pread");return 1;}
    for(size_t i=0;i<E;++i) emb[i]=ds4f_f16_to_f32(emb16[i]);
    stats("embedding",emb,E);
    for(int h=0;h<4;++h)memcpy(res+(size_t)h*E,emb,E*4);
    ds4f_rms_norm(flat,res,NULL,E*HC,1e-6f);
    const ds4f_tensor *fn=need(&g,"blk.0.hc_attn_fn.weight");
    if(ds4f_matvec(&g,fn,flat,mix)){perror("hc matvec");return 1;}
    float *scale=NULL,*base=NULL;load(&g,need(&g,"blk.0.hc_attn_scale.weight"),(void**)&scale);load(&g,need(&g,"blk.0.hc_attn_base.weight"),(void**)&base);
    printf("hc scale=%.7g %.7g %.7g base=%.7g %.7g %.7g %.7g\n",scale[0],scale[1],scale[2],base[0],base[1],base[2],base[3]);
    hc_split(split,mix,scale,base);
    float *cur=malloc(E*4);weighted(cur,res,split);stats("attn_pre",cur,E);
    float *norm=malloc(E*4);float *nw=NULL;load(&g,need(&g,"blk.0.attn_norm.weight"),(void**)&nw);ds4f_rms_norm(norm,cur,nw,E,1e-6f);
    float *qr=malloc(QR*4),*qrn=malloc(QR*4),*q=malloc(Q*4),*kv=malloc(HD*4),*kvn=malloc(HD*4),*w=NULL;
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_q_a.weight"),norm,qr))return 1;float *qan=NULL;load(&g,need(&g,"blk.0.attn_q_a_norm.weight"),(void**)&qan);ds4f_rms_norm(qrn,qr,qan,QR,1e-6f);
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_q_b.weight"),qrn,q))return 1;
    for(int h=0;h<64;++h)ds4f_rms_norm(q+(size_t)h*HD,q+(size_t)h*HD,NULL,HD,1e-6f);
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_kv.weight"),norm,kv))return 1;float *kvan=NULL;load(&g,need(&g,"blk.0.attn_kv_a_norm.weight"),(void**)&kvan);ds4f_rms_norm(kvn,kv,kvan,HD,1e-6f);stats("q",q,Q);stats("kv",kvn,HD);
    float *heads=malloc(Q*4);float *sinks=NULL;load(&g,need(&g,"blk.0.attn_sinks.weight"),(void**)&sinks);float inv=sqrtf(1.0f/512.0f);
    for(int h=0;h<64;++h){float score=0;for(size_t i=0;i<HD;++i)score+=q[(size_t)h*HD+i]*kvn[i];score*=inv;float a=expf(score-fmaxf(score,sinks[h])),b=expf(sinks[h]-fmaxf(score,sinks[h]));float wt=a/(a+b);for(size_t i=0;i<HD;++i)heads[(size_t)h*HD+i]=kvn[i]*wt;}
    float *low=calloc(8192,4),*aout=malloc(E*4),*after=malloc(E*HC*4);
    if(ds4f_matvec_group(&g,need(&g,"blk.0.attn_output_a.weight"),heads,8,4096,1024,low))return 1;
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_output_b.weight"),low,aout))return 1;
    hc_post(after,aout,res,split+4,split+8);stats("after_attn_hc",after,E*HC);
    float *final=malloc(E*HC*4);moe_layer0(&g,after,token,final);
    free(final);free(w);free(sinks);free(after);free(aout);free(low);free(kvn);free(kv);free(q);free(qrn);free(qr);free(qan);free(nw);free(norm);free(cur);free(base);free(scale);free(split);free(mix);free(flat);free(res);free(emb);free(emb16);ds4f_gguf_close(&g);return 0;
}

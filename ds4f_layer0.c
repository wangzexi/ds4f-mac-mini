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

int main(int argc,char **argv){
    if(argc<2){fprintf(stderr,"usage: %s MODEL.gguf [token]\n",argv[0]);return 2;}
    int token=argc>2?atoi(argv[2]):67;
    ds4f_gguf g; if(ds4f_gguf_open(&g,argv[1])){perror("GGUF");return 1;}
    const size_t E=4096, HC=4, HD=512, Q=64*512, QR=1024;
    float *emb=malloc(E*4), *res=malloc(E*HC*4), *flat=malloc(E*HC*4), *mix=malloc(24*4), *split=malloc(24*4);
    const ds4f_tensor *te=need(&g,"token_embd.weight");
    if(ds4f_gguf_read(&g,te,(uint64_t)token*E*2,emb,E*2)){perror("embedding pread");return 1;}
    for(size_t i=0;i<E;++i){uint16_t h;memcpy(&h,(uint8_t*)emb+i*2,2);emb[i]=ds4f_f16_to_f32(h);}
    for(int h=0;h<4;++h)memcpy(res+(size_t)h*E,emb,E*4);
    ds4f_rms_norm(flat,res,NULL,E*HC,1e-5f);
    const ds4f_tensor *fn=need(&g,"blk.0.hc_attn_fn.weight");
    if(ds4f_matvec(&g,fn,flat,mix)){perror("hc matvec");return 1;}
    float *scale=NULL,*base=NULL;load(&g,need(&g,"blk.0.hc_attn_scale.weight"),(void**)&scale);load(&g,need(&g,"blk.0.hc_attn_base.weight"),(void**)&base);
    hc_split(split,mix,scale,base);
    float *cur=malloc(E*4);weighted(cur,res,split);stats("attn_pre",cur,E);
    float *norm=malloc(E*4);float *nw=NULL;load(&g,need(&g,"blk.0.attn_norm.weight"),(void**)&nw);ds4f_rms_norm(norm,cur,nw,E,1e-5f);
    float *qr=malloc(QR*4),*qrn=malloc(QR*4),*q=malloc(Q*4),*kv=malloc(HD*4),*kvn=malloc(HD*4),*w=NULL;
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_q_a.weight"),norm,qr))return 1;float *qan=NULL;load(&g,need(&g,"blk.0.attn_q_a_norm.weight"),(void**)&qan);ds4f_rms_norm(qrn,qr,qan,QR,1e-5f);
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_q_b.weight"),qrn,q))return 1;
    for(int h=0;h<64;++h)ds4f_rms_norm(q+(size_t)h*HD,q+(size_t)h*HD,NULL,HD,1e-5f);
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_kv.weight"),norm,kv))return 1;float *kvan=NULL;load(&g,need(&g,"blk.0.attn_kv_a_norm.weight"),(void**)&kvan);ds4f_rms_norm(kvn,kv,kvan,HD,1e-5f);stats("q",q,Q);stats("kv",kvn,HD);
    float *heads=malloc(Q*4);float *sinks=NULL;load(&g,need(&g,"blk.0.attn_sinks.weight"),(void**)&sinks);float inv=sqrtf(1.0f/512.0f);
    for(int h=0;h<64;++h){float score=0;for(size_t i=0;i<HD;++i)score+=q[(size_t)h*HD+i]*kvn[i];score*=inv;float a=expf(score-fmaxf(score,sinks[h])),b=expf(sinks[h]-fmaxf(score,sinks[h]));float wt=a/(a+b);for(size_t i=0;i<HD;++i)heads[(size_t)h*HD+i]=kvn[i]*wt;}
    float *low=calloc(8192,4),*aout=malloc(E*4),*after=malloc(E*HC*4);
    if(ds4f_matvec_group(&g,need(&g,"blk.0.attn_output_a.weight"),heads,8,4096,1024,low))return 1;
    if(ds4f_matvec(&g,need(&g,"blk.0.attn_output_b.weight"),low,aout))return 1;
    hc_post(after,aout,res,split+4,split+8);stats("after_attn_hc",after,E*HC);
    free(w);free(sinks);free(after);free(aout);free(low);free(kvn);free(kv);free(q);free(qrn);free(qr);free(qan);free(nw);free(norm);free(cur);free(base);free(scale);free(split);free(mix);free(flat);free(res);free(emb);ds4f_gguf_close(&g);return 0;
}

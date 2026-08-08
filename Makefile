CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11

DS4F_FAST_CORE_OBJS = reference-ds4/ds4.o reference-ds4/ds4_distributed.o reference-ds4/ds4_tp.o reference-ds4/ds4_ssd.o reference-ds4/ds4_metal.o reference-ds4/ds4_layer_pack.o
.PHONY: all metal fast dspark clean ds4f-fast-reference
all: ds4f-probe ds4f-layer0 ds4f-first-token ds4f-tokenize ds4f-generate

metal: ds4f-generate-metal ds4f-first-token-metal

fast: ds4f-fast
dspark: ds4f-dspark-probe

ds4f-probe: src/ds4f_probe.o src/ds4f_gguf.o
	$(CC) $(CFLAGS) -o $@ $^

ds4f-layer0: src/ds4f_layer0.o src/ds4f_quant.o src/ds4f_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

ds4f-first-token: src/ds4f_first.o src/ds4f_quant.o src/ds4f_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

ds4f-first-token-metal: src/ds4f_first.o src/ds4f_quant_metal.o src/ds4f_gguf.o src/ds4f_metal.o
	$(CC) $(CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm

ds4f-tokenize: src/ds4f_tokenize.o src/ds4f_tokenizer.o src/ds4f_gguf.o
	$(CC) $(CFLAGS) -o $@ $^

ds4f-dspark-probe: src/ds4f_dspark_probe.o src/ds4f_gguf.o
	$(CC) $(CFLAGS) -o $@ $^

ds4f-generate: src/ds4f_generate.o src/ds4f_tokenizer.o src/ds4f_gguf.o src/ds4f_quant.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

ds4f-fast: src/ds4f_fast.o ds4f-fast-reference
	$(CC) $(CFLAGS) -o $@ src/ds4f_fast.o $(DS4F_FAST_CORE_OBJS) -framework Foundation -framework Metal -lm -pthread

ds4f-fast-reference:
	$(MAKE) -C reference-ds4 ds4
ds4f-generate-metal: src/ds4f_generate.o src/ds4f_tokenizer.o src/ds4f_gguf.o src/ds4f_quant_metal.o src/ds4f_metal.o src/ds4f_attention_metal.o
	$(CC) $(CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm

src/ds4f_fast.o: src/ds4f_fast.c reference-ds4/ds4.h
	$(CC) $(CFLAGS) -Ireference-ds4 -c -o $@ $<

src/ds4f_attention_metal.o: src/ds4f_attention_metal.m
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

src/ds4f_quant_metal.o: src/ds4f_quant.c src/ds4f_gguf.h src/ds4f_quant.h
	$(CC) $(CFLAGS) -DDS4F_USE_METAL -c -o $@ $<
src/ds4f_metal.o: src/ds4f_metal.m src/ds4f_gguf.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

src/%.o: src/%.c src/ds4f_gguf.h
	@mkdir -p src
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f ds4f-probe ds4f-layer0 ds4f-first-token ds4f-first-token-metal ds4f-tokenize ds4f-generate ds4f-generate-metal ds4f-dspark-probe ds4f-fast src/*.o

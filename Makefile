CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11

DS4F_FAST_CORE_OBJS = reference-ds4/ds4.o reference-ds4/ds4_distributed.o reference-ds4/ds4_tp.o reference-ds4/ds4_ssd.o reference-ds4/ds4_metal.o reference-ds4/ds4_layer_pack.o
.PHONY: all metal fast check-production dspark clean ds4f-fast-reference
all: ds4f-probe ds4f-layer0 ds4f-first-token ds4f-tokenize ds4f-generate

metal: ds4f-generate-metal ds4f-first-token-metal

fast: ds4f-fast ds4f-reuse ds4f-speed
check-production: ds4f-fast
	@test -n "$(MODEL)" || { echo "usage: make check-production MODEL=/path/to/Flash-0731.gguf" >&2; exit 2; }
	./scripts/check-production-regression.sh ./ds4f-fast "$(MODEL)"

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

ds4f-reuse: src/ds4f_reuse.o ds4f-fast-reference
	$(CC) $(CFLAGS) -o $@ src/ds4f_reuse.o $(DS4F_FAST_CORE_OBJS) -framework Foundation -framework Metal -lm -pthread

ds4f-fast-reference:
	$(MAKE) -C reference-ds4 ds4
ds4f-generate-metal: src/ds4f_generate.o src/ds4f_tokenizer.o src/ds4f_gguf.o src/ds4f_quant_metal.o src/ds4f_metal.o src/ds4f_attention_metal.o
	$(CC) $(CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm

src/ds4f_reuse.o: src/ds4f_reuse.c reference-ds4/ds4.h
	$(CC) $(CFLAGS) -Ireference-ds4 -c -o $@ $<

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
	rm -f ds4f-probe ds4f-layer0 ds4f-first-token ds4f-first-token-metal ds4f-tokenize ds4f-generate ds4f-generate-metal ds4f-dspark-probe ds4f-fast ds4f-reuse src/*.o ds4f-speed src/ds4f_fast_speed.o

DS4F_SPEED_CORE_OBJS = speed-ds4/ds4.o speed-ds4/ds4_distributed.o speed-ds4/ds4_tp.o speed-ds4/ds4_ssd.o speed-ds4/ds4_metal.o speed-ds4/ds4_layer_pack.o

ds4f-speed: src/ds4f_fast_speed.o ds4f-speed-reference
	$(CC) $(CFLAGS) -o $@ src/ds4f_fast_speed.o $(DS4F_SPEED_CORE_OBJS) -framework Foundation -framework Metal -lm -pthread

ds4f-speed-reference: scripts/prepare-speed-engine.sh
	./scripts/prepare-speed-engine.sh
	$(MAKE) -C speed-ds4 ds4

src/ds4f_fast_speed.o: src/ds4f_fast.c ds4f-speed-reference
	$(CC) $(CFLAGS) -DDS4F_SPEED_BUILD -Ispeed-ds4 -c -o $@ $<

CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11
RUNTIME_CFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99
RUNTIME_OBJCFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -fobjc-arc

DS4F_RUNTIME_CORE_OBJS = runtime/ds4.o runtime/ds4_distributed.o runtime/ds4_tp.o runtime/ds4_ssd.o runtime/ds4_metal.o runtime/ds4_layer_pack.o
DS4F_SERVER_OBJS = runtime/ds4_server.o runtime/ds4_help.o runtime/ds4_kvstore.o runtime/rax.o runtime/ds4_gpu_args.o $(DS4F_RUNTIME_CORE_OBJS)
.PHONY: all metal fast q4-fast server check-production dspark clean
all: ds4f-probe ds4f-layer0 ds4f-first-token ds4f-tokenize ds4f-generate

metal: ds4f-generate-metal ds4f-first-token-metal

fast: ds4f-q4-speed ds4f-server
q4-fast: ds4f-q4-speed ds4f-server
server: ds4f-server
check-production: ds4f-q4-speed
	@test -n "$(MODEL)" || { echo "usage: make check-production MODEL=/path/to/Flash-0731.gguf" >&2; exit 2; }
	./scripts/check-production-regression.sh ./ds4f-q4-speed "$(MODEL)"

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

ds4f-q4-speed: src/ds4f_fast_speed.o $(DS4F_RUNTIME_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm -pthread

ds4f-server: $(DS4F_SERVER_OBJS)
	$(CC) $(RUNTIME_CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm -pthread

runtime/ds4.o: runtime/ds4.c runtime/ds4.h runtime/ds4_ssd.h runtime/ds4_distributed.h runtime/ds4_gpu.h
	$(CC) $(RUNTIME_CFLAGS) -Iruntime -c -o $@ $<

runtime/ds4_server.o: runtime/ds4_server.c runtime/ds4.h runtime/ds4_ssd.h runtime/ds4_distributed.h runtime/ds4_help.h runtime/ds4_kvstore.h runtime/rax.h
	$(CC) $(RUNTIME_CFLAGS) -Iruntime -c -o $@ $<

runtime/ds4_metal.o: runtime/ds4_metal.m runtime/ds4_gpu.h $(wildcard runtime/metal/*.metal)
	$(CC) $(RUNTIME_OBJCFLAGS) -Iruntime -c -o $@ $<

runtime/%.o: runtime/%.c
	$(CC) $(RUNTIME_CFLAGS) -Iruntime -c -o $@ $<

src/ds4f_fast_speed.o: src/ds4f_fast.c runtime/ds4.h
	$(CC) $(CFLAGS) -DDS4F_SPEED_BUILD -Iruntime -c -o $@ $<

ds4f-generate-metal: src/ds4f_generate.o src/ds4f_tokenizer.o src/ds4f_gguf.o src/ds4f_quant_metal.o src/ds4f_metal.o src/ds4f_attention_metal.o
	$(CC) $(CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm

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
	rm -f ds4f-probe ds4f-layer0 ds4f-first-token ds4f-first-token-metal ds4f-tokenize ds4f-generate ds4f-generate-metal ds4f-dspark-probe ds4f-q4-speed ds4f-server src/*.o runtime/*.o

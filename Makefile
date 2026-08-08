CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11

all: ds4f-probe ds4f-layer0 ds4f-first-token ds4f-tokenize ds4f-generate

metal: ds4f-generate-metal ds4f-first-token-metal

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
	rm -f ds4f-probe ds4f-layer0 ds4f-first-token ds4f-first-token-metal ds4f-tokenize ds4f-generate ds4f-generate-metal ds4f-dspark-probe src/*.o

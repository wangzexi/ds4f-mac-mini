CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11
RUNTIME_CFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99
RUNTIME_OBJCFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -fobjc-arc

RUNTIME_CORE_OBJS = \
	runtime/ds4.o \
	runtime/ds4_distributed.o \
	runtime/ds4_tp.o \
	runtime/ds4_ssd.o \
	runtime/ds4_metal.o \
	runtime/ds4_layer_pack.o
SERVER_OBJS = \
	runtime/ds4_server.o \
	runtime/ds4_help.o \
	runtime/ds4_kvstore.o \
	runtime/rax.o \
	runtime/ds4_gpu_args.o \
	$(RUNTIME_CORE_OBJS)

.PHONY: all runner server check-production clean

all: runner server

runner: ds4f-q4-speed

server: ds4f-server

check-production: ds4f-q4-speed
	@test -n "$(MODEL)" || { echo "usage: make check-production MODEL=/path/to/Flash-0731.gguf" >&2; exit 2; }
	./scripts/check-production-regression.sh ./ds4f-q4-speed "$(MODEL)"

ds4f-q4-speed: src/ds4f_fast.o $(RUNTIME_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm -pthread

ds4f-server: $(SERVER_OBJS)
	$(CC) $(RUNTIME_CFLAGS) -o $@ $^ -framework Foundation -framework Metal -lm -pthread

runtime/ds4.o: runtime/ds4.c runtime/ds4.h runtime/ds4_ssd.h runtime/ds4_distributed.h runtime/ds4_gpu.h
	$(CC) $(RUNTIME_CFLAGS) -Iruntime -c -o $@ $<

runtime/ds4_server.o: runtime/ds4_server.c runtime/ds4.h runtime/ds4_ssd.h runtime/ds4_distributed.h runtime/ds4_help.h runtime/ds4_kvstore.h runtime/rax.h
	$(CC) $(RUNTIME_CFLAGS) -Iruntime -c -o $@ $<

runtime/ds4_metal.o: runtime/ds4_metal.m runtime/ds4_gpu.h $(wildcard runtime/metal/*.metal)
	$(CC) $(RUNTIME_OBJCFLAGS) -Iruntime -c -o $@ $<

runtime/%.o: runtime/%.c
	$(CC) $(RUNTIME_CFLAGS) -Iruntime -c -o $@ $<

src/ds4f_fast.o: src/ds4f_fast.c runtime/ds4.h
	$(CC) $(CFLAGS) -Iruntime -c -o $@ $<

clean:
	rm -f ds4f-q4-speed ds4f-server src/*.o runtime/*.o

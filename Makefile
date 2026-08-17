CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11
RUNTIME_CFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -ffunction-sections -fdata-sections
RUNTIME_OBJCFLAGS ?= -O3 -ffast-math -g -mcpu=native -Wall -Wextra -fobjc-arc -ffunction-sections -fdata-sections

BUILDDIR := build
QUANTS_DIR ?= reference-ds4/gguf-tools
QUANTIZER ?= $(BUILDDIR)/deepseek4-quantize
QUANTIZER_SRC := agents/skills/ds4f-mac-mini-ops/src/deepseek4-quantize.c

RUNTIME_CORE_OBJS = \
	$(BUILDDIR)/src/runtime/ds4.o \
	$(BUILDDIR)/src/runtime/ds4_distributed.o \
	$(BUILDDIR)/src/runtime/ds4_tp.o \
	$(BUILDDIR)/src/runtime/ds4_ssd.o \
	$(BUILDDIR)/src/runtime/ds4_metal.o \
	$(BUILDDIR)/src/runtime/ds4_layer_pack.o
SERVER_OBJS = \
	$(BUILDDIR)/src/runtime/ds4_server.o \
	$(BUILDDIR)/src/runtime/ds4_help.o \
	$(BUILDDIR)/src/runtime/ds4_kvstore.o \
	$(BUILDDIR)/src/runtime/rax.o \
	$(BUILDDIR)/src/runtime/ds4_gpu_args.o \
	$(RUNTIME_CORE_OBJS)

.PHONY: all server quantizer clean

all: ds4f-q4-speed ds4f-server

server: ds4f-server

quantizer: $(QUANTIZER)

$(QUANTIZER): $(QUANTIZER_SRC) | $(BUILDDIR)
	@test -f "$(QUANTS_DIR)/quants.c" && test -f "$(QUANTS_DIR)/quants.h" || { echo "missing quants.c/quants.h; see agents/skills/ds4f-mac-mini-ops/SKILL.md for the pinned upstream checkout" >&2; exit 2; }
	$(CC) -O2 -Wall -Wextra -Werror -std=c11 -I"$(QUANTS_DIR)" -o $@ $(QUANTIZER_SRC) "$(QUANTS_DIR)/quants.c" -lm -pthread

ds4f-q4-speed: $(BUILDDIR)/src/ds4f.o $(RUNTIME_CORE_OBJS)
	$(CC) $(CFLAGS) -Wl,-dead_strip -o $@ $^ -framework Foundation -framework Metal -lm -pthread

ds4f-server: $(SERVER_OBJS)
	$(CC) $(RUNTIME_CFLAGS) -Wl,-dead_strip -o $@ $^ -framework Foundation -framework Metal -lm -pthread

$(BUILDDIR)/src/runtime/ds4.o: src/runtime/ds4.c src/runtime/ds4.h src/runtime/ds4_ssd.h src/runtime/ds4_distributed.h src/runtime/ds4_gpu.h | $(BUILDDIR)/src/runtime
	$(CC) $(RUNTIME_CFLAGS) -Isrc/runtime -c -o $@ $<

$(BUILDDIR)/src/runtime/ds4_server.o: src/runtime/ds4_server.c src/runtime/ds4.h src/runtime/ds4_ssd.h src/runtime/ds4_distributed.h src/runtime/ds4_help.h src/runtime/ds4_kvstore.h src/runtime/rax.h | $(BUILDDIR)/src/runtime
	$(CC) $(RUNTIME_CFLAGS) -Isrc/runtime -c -o $@ $<

$(BUILDDIR)/src/runtime/ds4_metal.o: src/runtime/ds4_metal.m src/runtime/ds4_gpu.h $(wildcard src/runtime/metal/*.metal) | $(BUILDDIR)/src/runtime
	$(CC) $(RUNTIME_OBJCFLAGS) -Isrc/runtime -c -o $@ $<

$(BUILDDIR)/src/runtime/%.o: src/runtime/%.c | $(BUILDDIR)/src/runtime
	$(CC) $(RUNTIME_CFLAGS) -Isrc/runtime -c -o $@ $<

$(BUILDDIR)/src/ds4f.o: src/ds4f.c src/runtime/ds4.h | $(BUILDDIR)/src
	$(CC) $(CFLAGS) -Isrc/runtime -c -o $@ $<

$(BUILDDIR)/src/runtime $(BUILDDIR)/src:
	mkdir -p $@

$(BUILDDIR):
	mkdir -p $@

clean:
	rm -rf $(BUILDDIR) ds4f-q4-speed ds4f-server

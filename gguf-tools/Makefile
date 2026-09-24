CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wshadow -Wno-unused-function
LDFLAGS ?=
LDLIBS ?= -lm -pthread

all: qwen36-quantize

test: qwen36-quantize
	python3 tests/test_strip_nextn.py ./qwen36-quantize

qwen36-quantize: qwen36-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) -o $@ qwen36-quantize.c quants.c $(LDFLAGS) $(LDLIBS)

clean:
	rm -f qwen36-quantize

.PHONY: all clean test

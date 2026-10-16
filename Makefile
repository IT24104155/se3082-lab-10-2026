NVCC ?= nvcc
CUDAFLAGS ?= -O2
SOURCES = $(wildcard Exercise01/*.cu Exercise02/*.cu Exercise03/*.cu Exercise04/*.cu)
PROGRAMS = $(addprefix bin/,$(notdir $(SOURCES:.cu=)))
vpath %.cu Exercise01 Exercise02 Exercise03 Exercise04

.PHONY: all run clean
all: $(PROGRAMS)
bin:
	mkdir -p bin
bin/%: %.cu | bin
	$(NVCC) $(CUDAFLAGS) $< -o $@
run: all
	set -e; for program in $(PROGRAMS); do ./$$program; done
clean:
	rm -rf bin

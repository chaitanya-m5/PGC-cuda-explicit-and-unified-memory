# Theme 6: Unified Memory vs Explicit Memory Copy
#
#   make          build the three executables into build/
#   make run      build, run all three, write results/results.csv
#   make plot     regenerate the graphs from results/results.csv
#   make clean    remove build artefacts
#
# If -arch=native is not supported by your nvcc, set the architecture:
#   make ARCH=-arch=sm_120        (example: RTX 50-series)

NVCC  ?= nvcc
ARCH  ?= -arch=native
FLAGS ?= -O2 -std=c++14
RUNS  ?= 5
BIN    = build
CSV    = results/results.csv
HEADER = Dataset,Method,H2D_ms,Kernel_ms,D2H_ms,Prefetch_ms,Total_ms,EndToEnd_ms,Correctness

all: $(BIN)/explicit $(BIN)/unified $(BIN)/prefetch

$(BIN)/%: src/%.cu src/common.cuh
	@mkdir -p $(BIN)
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $<

run: all
	@mkdir -p results
	@echo "$(HEADER)" > $(CSV)
	./$(BIN)/explicit $(CSV) $(RUNS)
	./$(BIN)/unified  $(CSV) $(RUNS)
	./$(BIN)/prefetch $(CSV) $(RUNS)
	@echo "Results written to $(CSV)"

plot:
	python3 scripts/plot_results.py

clean:
	rm -rf $(BIN)

.PHONY: all run plot clean

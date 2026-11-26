# Data Provenance

## Source of `results/results.csv`

Transcribed from the terminal screenshot
`screenshots/terminal/make_run_and_results_csv_RTX5060Ti.png`
(`make run` followed by `cat results/results.csv`, GPU: NVIDIA GeForce RTX 5060 Ti), produced by an earlier build of the program that is not part of this repository.

Other screenshots were used only for cross-checking:

| Screenshot | Role |
|---|---|
| `first_run_csv_and_plot_script.png` | First run; 6-row CSV (Explicit and Unified only) |
| `run_unified_memory_experiment_earlier_build.png` | Earlier build with different prefetch accounting; not used for the CSV |
| `legacy_total_vs_kernel_only_graph.png` | Original graph; values do not match the final CSV |

## Correction applied: 16M explicit row

| Source | Kernel_ms | D2H_ms | Total_ms |
|---|---:|---:|---:|
| `make_run...` screenshot (as captured) | 4.61343 | 0.527456 | 16.9234 |
| `first_run...` screenshot | 0.527456 | 4.61343 | 16.9234 |
| **Used in `results.csv`** | **0.527456** | **4.61343** | **16.9234** |

Reason: the two values appear swapped in the later screenshot. Assuming 4-byte elements:

| Check | As captured | Used |
|---|---|---|
| Kernel throughput (192 MB moved) | 192 MB / 4.61 ms = 41.6 GB/s (inconsistent with 64M run) | 192 MB / 0.527 ms = 364 GB/s (matches 64M: 768 MB / 2.111 ms = 363.8 GB/s) |
| D2H rate (64 MB) | 64 MB / 0.527 ms = 121 GB/s (beyond PCIe) | 64 MB / 4.61 ms = 13.9 GB/s (matches 64M: 15.2 GB/s) |

The earlier-build screenshot also shows 16M D2H = 16.9234 ms with total 33.3193 ms, which is the total of the other runs mislabelled as D2H.

## Not measured in the captured run

- `EndToEnd_ms` (recorded as `NA`).
- Run-to-run variation.
- CPU model, RAM, driver version, CUDA version.

Run `make run` with the sources in `src/` to produce a fully measured CSV.

# Methodology

## Common to all approaches

- Kernel: `C[i] = A[i] + B[i]`, 256 threads per block, one thread per element.
- Data type: `float`; `A[i] = i % 1024`, `B[i] = (3 * i) % 1024` (exactly representable sums).
- Dataset sizes: 1,000,000, 16,000,000 and 64,000,000 elements.
- Verification: every element compared with `A[i] + B[i]` using exact equality (not timed).
- Repetitions: 5 per dataset by default (`make run RUNS=n`), each with fresh allocations; the CSV stores the mean.
- Warm-up: context creation and one dummy kernel launch before the first measurement.
- Stage timing: CUDA events on the default stream. End-to-end timing: host wall-clock (`steady_clock`).

## Timing boundaries

| Approach | H2D_ms | Kernel_ms | D2H_ms | Prefetch_ms | Total_ms | EndToEnd_ms |
|---|---|---|---|---|---|---|
| Explicit | two `cudaMemcpy` H2D | kernel launch to completion | `cudaMemcpy` D2H | 0 | H2D + kernel + D2H | before first copy until D2H done |
| Unified | 0 | kernel, **including on-demand migration** | 0 | 0 | kernel only | before kernel until one element per 4 KiB page of C has been read on the CPU |
| Prefetch | 0 | kernel | 0 | prefetch of A, B, C to GPU | prefetch + kernel | before first prefetch until C has been prefetched back and read on the CPU |

Excluded from every timing: allocation, CPU initialisation of inputs, verification, deallocation.

`Total_ms` is a **measured-region** total and is not equivalent across approaches. `EndToEnd_ms` uses one common boundary (first transfer or launch until the result is readable on the CPU) and is the column to use for a like-for-like ranking.

## Fairness notes

- Explicit copies use pageable host memory. Pinned memory (`cudaMallocHost`) would speed them up.
- In the Unified version, the result is touched on the CPU after the kernel only inside the end-to-end measurement.
- Results on WSL2 may differ from native Linux because managed-memory support differs.

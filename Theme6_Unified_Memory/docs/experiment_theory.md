# Experiment Theory

## Memory spaces

| Term | Description |
|---|---|
| Host memory | System RAM attached to the CPU; input data normally starts here |
| Device memory | GPU on-board memory; kernels access it at high bandwidth |
| PCIe | The CPU-GPU link; copies and page migrations are limited by its bandwidth |

## Memory management strategies

| Strategy | Mechanism | Programmer effort | Where transfer cost appears |
|---|---|---|---|
| Explicit copy | `cudaMalloc` + `cudaMemcpy` | High | Separate H2D and D2H phases |
| Unified Memory | `cudaMallocManaged` | Low | Inside the kernel (GPU page faults) and on CPU access |
| Unified Memory + prefetch | `cudaMallocManaged` + `cudaMemPrefetchAsync` | Medium | Separate prefetch phase before the kernel |

## Page migration

Managed memory is divided into pages. If the GPU accesses a page resident in host memory, a fault is raised and the driver migrates the page to the GPU before the access completes. Faults are serviced while the kernel runs, so the kernel time includes migration time.

## Prefetching

`cudaMemPrefetchAsync(ptr, bytes, device)` requests migration of a managed range to a processor before use. Migration is done in bulk, which typically uses the link more efficiently than many individual faults.

## Transfer-overhead formula

```
Transfer overhead (%) = (H2D + D2H) / Total * 100
```

## Kernel memory throughput

For vector addition each element reads two floats and writes one:

```
throughput = 3 * N * 4 bytes / kernel_time
```

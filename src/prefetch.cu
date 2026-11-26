// prefetch.cu - Approach 3: Unified Memory with prefetching
// cudaMallocManaged + cudaMemPrefetchAsync: data is moved to the GPU in bulk
// BEFORE the kernel runs, so the kernel should not take page faults.
//
// Usage: ./prefetch [results.csv] [runs]
#include "common.cuh"

static Result runOnce(size_t n, int device) {
    const size_t bytes = n * sizeof(float);
    Result r;

    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    initInputs(a, b, n);  // first touch on the CPU (not timed)

    GpuTimer timer;
    const double wall0 = nowMs();

    // 1) Prefetch inputs and output to the GPU (bulk migration).
    timer.start();
    prefetchManaged(a, bytes, device);
    prefetchManaged(b, bytes, device);
    prefetchManaged(c, bytes, device);
    r.prefetch_ms = timer.stop();

    // 2) Kernel (data already resident on the GPU).
    timer.start();
    vectorAdd<<<blocksFor(n), kThreadsPerBlock>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = timer.stop();

    r.total_ms = r.prefetch_ms + r.kernel_ms;  // measured-region total

    // 3) Prefetch the result back to the CPU (counted in end-to-end only).
    prefetchManaged(c, bytes, cudaCpuDeviceId);
    CUDA_CHECK(cudaDeviceSynchronize());
    volatile float sink = touchPages(c, n);
    (void)sink;
    r.end_to_end_ms = nowMs() - wall0;

    r.correct = verifyResult(a, b, c, n);  // not timed

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return r;
}

int main(int argc, char** argv) {
    const char* csv = (argc > 1) ? argv[1] : nullptr;
    const int runs = parseRuns(argc, argv);
    printDeviceInfo(true);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    warmUpDevice();

    for (int s = 0; s < kNumSizes; ++s) {
        Result all[kMaxRuns];
        for (int i = 0; i < runs; ++i) all[i] = runOnce(kSizes[s], device);
        Result m = averageResults(all, runs);
        std::printf("[Prefetch %-3s] Prefetch %9.5f | Kernel %9.5f | "
                    "Total %9.5f | End-to-end %9.5f ms | %s\n",
                    kSizeLabels[s], m.prefetch_ms, m.kernel_ms, m.total_ms,
                    m.end_to_end_ms, m.correct ? "PASS" : "FAIL");
        appendCsv(csv, kSizes[s], "Prefetch", m);
        if (!m.correct) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

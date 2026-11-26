// unified.cu - Approach 2: Unified Memory (no prefetch)
// cudaMallocManaged: the same pointers are used on CPU and GPU. Pages are
// migrated on demand when the kernel (GPU page faults) or the CPU touches them.
//
// Usage: ./unified [results.csv] [runs]
#include "common.cuh"

static Result runOnce(size_t n) {
    const size_t bytes = n * sizeof(float);
    Result r;

    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    initInputs(a, b, n);  // first touch on the CPU (not timed)

    GpuTimer timer;
    const double wall0 = nowMs();

    // Kernel: includes the on-demand migration of A, B and C to the GPU.
    timer.start();
    vectorAdd<<<blocksFor(n), kThreadsPerBlock>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());

    r.total_ms = r.kernel_ms;  // measured-region total = kernel only

    // Bring the result back to the CPU so that end-to-end is comparable with
    // the explicit version, which includes its D2H copy.
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
    warmUpDevice();

    for (int s = 0; s < kNumSizes; ++s) {
        Result all[kMaxRuns];
        for (int i = 0; i < runs; ++i) all[i] = runOnce(kSizes[s]);
        Result m = averageResults(all, runs);
        std::printf("[Unified  %-3s] Kernel %9.5f | Total %9.5f | "
                    "End-to-end %9.5f ms | %s\n",
                    kSizeLabels[s], m.kernel_ms, m.total_ms, m.end_to_end_ms,
                    m.correct ? "PASS" : "FAIL");
        appendCsv(csv, kSizes[s], "Unified", m);
        if (!m.correct) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

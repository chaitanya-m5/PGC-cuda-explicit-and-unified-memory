// explicit.cu - Approach 1: Explicit Memory Copy
// cudaMalloc + cudaMemcpy (H2D) -> kernel -> cudaMemcpy (D2H) + cudaFree.
//
// Usage: ./explicit [results.csv] [runs]
#include "common.cuh"

static Result runOnce(size_t n) {
    const size_t bytes = n * sizeof(float);
    Result r;

    // Host buffers (pageable memory) and initial data.
    float* h_a = static_cast<float*>(std::malloc(bytes));
    float* h_b = static_cast<float*>(std::malloc(bytes));
    float* h_c = static_cast<float*>(std::malloc(bytes));
    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::exit(EXIT_FAILURE);
    }
    initInputs(h_a, h_b, n);

    // Device buffers.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    GpuTimer timer;
    const double wall0 = nowMs();

    // 1) Host -> Device
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    r.h2d_ms = timer.stop();

    // 2) Kernel
    timer.start();
    vectorAdd<<<blocksFor(n), kThreadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    r.kernel_ms = timer.stop();

    // 3) Device -> Host
    timer.start();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    r.d2h_ms = timer.stop();

    r.end_to_end_ms = nowMs() - wall0;
    r.total_ms = r.h2d_ms + r.kernel_ms + r.d2h_ms;  // measured-region total
    r.correct = verifyResult(h_a, h_b, h_c, n);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    return r;
}

int main(int argc, char** argv) {
    const char* csv = (argc > 1) ? argv[1] : nullptr;
    const int runs = parseRuns(argc, argv);
    printDeviceInfo(false);
    warmUpDevice();

    for (int s = 0; s < kNumSizes; ++s) {
        Result all[kMaxRuns];
        for (int i = 0; i < runs; ++i) all[i] = runOnce(kSizes[s]);
        Result m = averageResults(all, runs);
        std::printf("[Explicit %-3s] H2D %9.5f | Kernel %9.5f | D2H %9.5f | "
                    "Total %9.5f | End-to-end %9.5f ms | %s\n",
                    kSizeLabels[s], m.h2d_ms, m.kernel_ms, m.d2h_ms, m.total_ms,
                    m.end_to_end_ms, m.correct ? "PASS" : "FAIL");
        appendCsv(csv, kSizes[s], "Explicit", m);
        if (!m.correct) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

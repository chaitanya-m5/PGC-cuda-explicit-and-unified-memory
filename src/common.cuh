// common.cuh - shared utilities for the Theme 6 experiment
// (Unified Memory vs Explicit Memory Copy, CUDA vector addition).
//
// Every program in this project uses the SAME kernel, the SAME input data
// pattern and the SAME verification routine, so that only the memory
// management strategy differs between the three experiments.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,   \
                         __LINE__, cudaGetErrorString(err__), #call);          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Experiment configuration
// ---------------------------------------------------------------------------
constexpr int kNumSizes = 3;
constexpr size_t kSizes[kNumSizes] = {1000000ULL, 16000000ULL, 64000000ULL};
constexpr const char* kSizeLabels[kNumSizes] = {"1M", "16M", "64M"};
constexpr int kThreadsPerBlock = 256;
constexpr int kDefaultRuns = 5;  // number of repetitions averaged per dataset
constexpr int kMaxRuns = 100;

// ---------------------------------------------------------------------------
// Kernel: C[i] = A[i] + B[i]
// ---------------------------------------------------------------------------
__global__ void vectorAdd(const float* a, const float* b, float* c, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

inline int blocksFor(size_t n) {
    return static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

// ---------------------------------------------------------------------------
// Input initialisation and result verification (CPU side)
// Values are small integers stored as float, so the sums are exactly
// representable and the check can use exact equality.
// ---------------------------------------------------------------------------
inline void initInputs(float* a, float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 1024);
        b[i] = static_cast<float>((i * 3) % 1024);
    }
}

inline bool verifyResult(const float* a, const float* b, const float* c,
                         size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (c[i] != a[i] + b[i]) return false;
    return true;
}

// Touch one element per 4 KiB page so that every page of a managed buffer is
// migrated back to host memory (used for the end-to-end measurement only).
inline float touchPages(const float* p, size_t n) {
    float sink = 0.0f;
    for (size_t i = 0; i < n; i += 1024) sink += p[i];
    return sink;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
// GPU-side timing with CUDA events (stream 0).
struct GpuTimer {
    cudaEvent_t start_{}, stop_{};
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    float stop() {  // returns elapsed milliseconds
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Host wall-clock timing (milliseconds).
inline double nowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// Result record. Columns follow results/results.csv.
//   total_ms     : measured-region total (definition differs per method, see
//                  docs/methodology.md)
//   end_to_end_ms: host wall-clock from before the first transfer/launch until
//                  the result is readable on the CPU (same definition for all
//                  three methods, hence directly comparable)
// ---------------------------------------------------------------------------
struct Result {
    double h2d_ms = 0, kernel_ms = 0, d2h_ms = 0, prefetch_ms = 0;
    double total_ms = 0, end_to_end_ms = 0;
    bool correct = true;
};

inline Result averageResults(const Result* r, int runs) {
    Result m;
    for (int i = 0; i < runs; ++i) {
        m.h2d_ms += r[i].h2d_ms;
        m.kernel_ms += r[i].kernel_ms;
        m.d2h_ms += r[i].d2h_ms;
        m.prefetch_ms += r[i].prefetch_ms;
        m.total_ms += r[i].total_ms;
        m.end_to_end_ms += r[i].end_to_end_ms;
        m.correct = m.correct && r[i].correct;
    }
    m.h2d_ms /= runs;
    m.kernel_ms /= runs;
    m.d2h_ms /= runs;
    m.prefetch_ms /= runs;
    m.total_ms /= runs;
    m.end_to_end_ms /= runs;
    return m;
}

// Append one CSV row. Header: see Makefile / results/results.csv.
inline void appendCsv(const char* path, size_t n, const char* method,
                      const Result& r) {
    if (!path) return;
    FILE* f = std::fopen(path, "a");
    if (!f) {
        std::fprintf(stderr, "Cannot open %s for appending\n", path);
        return;
    }
    std::fprintf(f, "%zu,%s,%g,%g,%g,%g,%g,%g,%s\n", n, method, r.h2d_ms,
                 r.kernel_ms, r.d2h_ms, r.prefetch_ms, r.total_ms,
                 r.end_to_end_ms, r.correct ? "PASS" : "FAIL");
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// Device preparation and capability report
// ---------------------------------------------------------------------------
// Initialises the CUDA context and runs one dummy launch so that one-time
// start-up cost is not charged to the first measured dataset.
inline void warmUpDevice() {
    CUDA_CHECK(cudaFree(0));
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, 256 * sizeof(float)));
    vectorAdd<<<1, 256>>>(d, d, d, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d));
}

inline void printDeviceInfo(bool needManaged) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int managed = 0, concurrent = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrent,
                                      cudaDevAttrConcurrentManagedAccess, 0));
    std::printf("GPU: %s | managed memory: %s | concurrent managed access: %s\n",
                prop.name, managed ? "yes" : "no", concurrent ? "yes" : "no");
    if (needManaged && !managed) {
        std::fprintf(stderr, "This device does not support managed memory.\n");
        std::exit(EXIT_FAILURE);
    }
}

// Parse the optional number of repetitions (argv[2]); default kDefaultRuns.
inline int parseRuns(int argc, char** argv) {
    if (argc > 2) {
        int r = std::atoi(argv[2]);
        if (r > 0 && r <= kMaxRuns) return r;
    }
    return kDefaultRuns;
}

// ---------------------------------------------------------------------------
// Prefetch wrapper: the cudaMemPrefetchAsync signature changed in CUDA 13.
// ---------------------------------------------------------------------------
inline void prefetchManaged(const void* ptr, size_t bytes, int device) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    if (device == cudaCpuDeviceId) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
    }
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0, 0));
#else
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, 0));
#endif
}

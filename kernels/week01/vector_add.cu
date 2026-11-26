// Week 1 practice — vector add
// Build & run (GPU node):  nvcc -O3 -arch=sm_90 --resource-usage -o vector_add vector_add.cu && ./vector_add
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ======================= YOUR CODE =======================
// TODO 1: each thread computes ONE element: c[i] = a[i] + b[i]
//         - compute the global index i
//         - add the bounds check
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
    
}
// =========================================================

// ================= STEP 3: divergence — YOUR CODE =================
// TODO 3a: even LANES compute a+b, odd LANES compute a*b
//          (branch on threadIdx.x % 2  → lanes of one warp disagree)
__global__ void add_lane_div(const float* a, const float* b, float* c, int n) {   // (AI-written)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (threadIdx.x % 2 == 0) c[i] = a[i] + b[i];   // even lanes
        else                      c[i] = a[i] * b[i];   // odd lanes  → same warp, two paths
    }
}

// TODO 3b: even WARPS compute a+b, odd WARPS compute a*b
//          (branch on (threadIdx.x / 32) % 2  → whole warps agree)
__global__ void add_warp_div(const float* a, const float* b, float* c, int n) {   // (AI-written)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((threadIdx.x / 32) % 2 == 0) c[i] = a[i] + b[i];   // even warps
        else                             c[i] = a[i] * b[i];   // odd warps → each warp takes one path
    }
}
// ==================================================================

// timing helper (boilerplate): median of 50 runs after 5 warm-ups, in microseconds
template <typename F> float time_us(F launch) {
    for (int w = 0; w < 5; w++) launch();
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int runs = 50; float ms[runs];
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(t0); launch(); cudaEventRecord(t1);
        cudaEventSynchronize(t1); cudaEventElapsedTime(&ms[r], t0, t1);
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    for (int x = 1; x < runs; x++) for (int y = x; y > 0 && ms[y] < ms[y-1]; y--) { float t = ms[y]; ms[y] = ms[y-1]; ms[y-1] = t; }
    return ms[runs / 2] * 1000.0f;
}

int main() {
    const int n = (1 << 24) + 7;          // deliberately NOT a multiple of the block size
    const size_t bytes = (size_t)n * sizeof(float);

    // host data
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { ha[i] = rand() / (float)RAND_MAX; hb[i] = rand() / (float)RAND_MAX; }

    // device data (lives in HBM = global memory)
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes)); CUDA_CHECK(cudaMalloc(&db, bytes)); CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dc, 0, bytes));

    // ======================= YOUR CODE =======================
    // TODO 2: choose block size and grid size, then launch `add`
    int block = 256;
    int grid  = (n + block - 1) / block;                      // must cover ALL n elements
    add<<<grid, block>>>(da, db, dc, n);
    // add<<<?, ?>>>(...);
    // =========================================================

    CUDA_CHECK(cudaGetLastError());        // launch errors (bad config etc.)
    CUDA_CHECK(cudaDeviceSynchronize());   // wait for the GPU; catches kernel errors

    // check against CPU
    CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; i++) if (fabsf(hc[i] - (ha[i] + hb[i])) > 1e-6f) { if (bad < 3) printf("mismatch at %d\n", i); bad++; }
    printf("n=%d block=%d grid=%d threads=%lld  -> %s (%d bad)\n",
           n, block, grid, (long long)grid * block, bad ? "FAIL" : "PASS", bad);

    // ================== STEP 2: timing sweep (boilerplate) ==================
    // Data = 3 arrays × 67 MB = 201 MB, much bigger than the 50 MB L2,
    // so every run really streams from HBM.
    printf("\nblock   grid     median_us   GB/s    %%of_peak(3350)\n");
    int sizes[] = {32, 64, 128, 256, 512, 1024};
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int s : sizes) {
        int g = (n + s - 1) / s;                        // your grid formula
        for (int w = 0; w < 5; w++) add<<<g, s>>>(da, db, dc, n);   // warm-up
        CUDA_CHECK(cudaDeviceSynchronize());

        const int runs = 50; float ms[runs];
        for (int r = 0; r < runs; r++) {
            cudaEventRecord(t0);
            add<<<g, s>>>(da, db, dc, n);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms[r], t0, t1);
        }
        CUDA_CHECK(cudaGetLastError());
        // median: sort the 50 times, take the middle
        for (int x = 1; x < runs; x++) for (int y = x; y > 0 && ms[y] < ms[y-1]; y--) { float t = ms[y]; ms[y] = ms[y-1]; ms[y-1] = t; }
        float med_us = ms[runs / 2] * 1000.0f;
        double gbps = 3.0 * bytes / (med_us * 1e-6) / 1e9;   // read a, read b, write c
        printf("%5d  %6d   %9.2f   %6.0f   %5.1f%%\n", s, g, med_us, gbps, 100.0 * gbps / 3350.0);
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    // ========================================================================

    // ================== STEP 3: divergence timing (boilerplate) ==================
    {
        int s = 256, g = (n + s - 1) / s;
        float t_plain = time_us([&] { add<<<g, s>>>(da, db, dc, n); });
        float t_lane  = time_us([&] { add_lane_div<<<g, s>>>(da, db, dc, n); });
        float t_warp  = time_us([&] { add_warp_div<<<g, s>>>(da, db, dc, n); });
        CUDA_CHECK(cudaGetLastError());
        printf("\nDivergence test (block 256):\n");
        printf("  plain add       %7.2f us\n", t_plain);
        printf("  lane-divergent  %7.2f us   (%.2fx)\n", t_lane, t_lane / t_plain);
        printf("  warp-divergent  %7.2f us   (%.2fx)\n", t_warp, t_warp / t_plain);
    }
    // =============================================================================

    cudaFree(da); cudaFree(db); cudaFree(dc); free(ha); free(hb); free(hc);
    return bad != 0;
}

#pragma once
#include <cstdio>
#include <cmath>

#include "error.h"

constexpr unsigned SKIP = 5, REPEATS = 5;
constexpr size_t M = 5120, N = 4096;
constexpr size_t real_size = sizeof(real);
constexpr size_t MN = M * N;
constexpr size_t MN_size = MN * real_size;

void transpose_matrix(const real *, real *);

void random_init(real *data, const size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        data[i] = real(rand()) / RAND_MAX;
    }
}

__global__ void check_kernel(const real (*A)[N], real (*B)[M])
{
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iy < M && ix < N) {
        B[ix][iy] = A[iy][ix];
    }
}

bool check(const real *A, real *B) {
    real *h_B = nullptr;
    CHECK(cudaMallocHost(&h_B, MN_size));

    real *d_A = nullptr, *d_B = nullptr;
    CHECK(cudaMalloc(&d_A, MN_size));
    CHECK(cudaMalloc(&d_B, MN_size));

    CHECK(cudaMemcpy(d_A, A, MN_size, cudaMemcpyHostToDevice));

    const real (*d_nA)[N] = reinterpret_cast<decltype(d_nA)>(d_A);
    real (*d_nB)[M] = reinterpret_cast<decltype(d_nB)>(d_B);

    dim3 block_size(32, 32);
    // N是列对应x，M是行对应y
    dim3 grid_size(DIVUP(N, block_size.x), DIVUP(M, block_size.y));
    check_kernel<<<grid_size, block_size>>>(d_nA, d_nB);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_B, d_B, MN_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            size_t pos = i * M + j;
            real res = h_B[pos], v = B[pos];
            if (std::fabs(res - v) > EPSILON) {
                printf("B[%u][%u] not match, %.15f vs %.15f\n", unsigned(j), unsigned(i), res, v);
                CHECK(cudaFreeHost(h_B));
                return false;
            }
        }
    }
    CHECK(cudaFreeHost(h_B));
    return true;
}

real timing(const real *A, real *B)
{
    float elapsed_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, 0));

    transpose_matrix(A, B);

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return elapsed_time;
}

void launch_cpu()
{
    real *h_A = nullptr, *h_B = nullptr;
    CHECK(cudaMallocHost(&h_A, MN_size));
    CHECK(cudaMallocHost(&h_B, MN_size));

    random_init(h_A, MN);

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(h_A, h_B);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(h_A, h_B);
        total_time += elapsed_time;
    }
    printf("Time: %9.3f ms\n", total_time / REPEATS);

    printf("Check: %s\n", check(h_A, h_B) ? "OK" : "Failed");

    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
}

void launch_gpu()
{
    real *h_A = nullptr, *h_B = nullptr;
    CHECK(cudaMallocHost(&h_A, MN_size));
    CHECK(cudaMallocHost(&h_B, MN_size));

    random_init(h_A, MN);

    real *d_A = nullptr, *d_B = nullptr;
    CHECK(cudaMalloc(&d_A, MN_size));
    CHECK(cudaMalloc(&d_B, MN_size));

    CHECK(cudaMemcpy(d_A, h_A, MN_size, cudaMemcpyHostToDevice));

    float elapsed_time = 0, total_time = 0;
    for (unsigned i = 0; i < SKIP; ++i) {
        elapsed_time = timing(d_A, d_B);
    }
    for (unsigned i = 0; i < REPEATS; ++i) {
        elapsed_time = timing(d_A, d_B);
        total_time += elapsed_time;
    }
    printf("Time: %9.3f ms\n", total_time / REPEATS);

    CHECK(cudaMemcpy(h_B, d_B, MN_size, cudaMemcpyDeviceToHost));
    printf("Check: %s\n", check(h_A, h_B) ? "OK" : "Failed");

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
}

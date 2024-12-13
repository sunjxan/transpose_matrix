#include "common.hpp"

// 朴素实现，注意iy和ix对行列的编码

__global__ void kernel(const real (*A)[N], real (*B)[M])
{
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (iy < M && ix < N) {
        B[ix][iy] = A[iy][ix];
    }
}

void transpose_matrix(const real *A, real *B)
{
    const real (*nA)[N] = reinterpret_cast<decltype(nA)>(A);
    real (*nB)[M] = reinterpret_cast<decltype(nB)>(B);

    dim3 block_size(32, 32);
    // N是列对应x，M是行对应y
    dim3 grid_size(DIVUP(N, block_size.x), DIVUP(M, block_size.y));
    kernel<<<grid_size, block_size>>>(nA, nB);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

int main()
{
    launch_gpu();
    return 0;
}
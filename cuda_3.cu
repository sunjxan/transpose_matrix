#include "common.hpp"

// 对角转置

__global__ void kernel(const real (*A)[N], real (*B)[M])
{

    unsigned int blk_y = blockIdx.y;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned iy = blk_y * blockDim.y + threadIdx.y;
    unsigned ix = blk_x * blockDim.x + threadIdx.x;

    if (iy < N && ix < M) {
        B[iy][ix] = A[ix][iy];
    }
}

void transpose_matrix(const real *A, real *B)
{
    const real (*nA)[N] = reinterpret_cast<decltype(nA)>(A);
    real (*nB)[M] = reinterpret_cast<decltype(nB)>(B);

    dim3 block_size(32, 32);
    // N是列对应y，M是行对应x
    dim3 grid_size(DIVUP(M, block_size.x), DIVUP(N, block_size.y));
    kernel<<<grid_size, block_size>>>(nA, nB);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

int main()
{
    launch_gpu();
    return 0;
}
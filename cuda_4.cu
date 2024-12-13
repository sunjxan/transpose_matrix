#include "common.hpp"

// 使用共享内存，读写都使用对齐合并访问全局内存

__global__ void kernel(const real (*A)[N], real (*B)[M])
{
    unsigned ty = threadIdx.y, iy = blockIdx.y * blockDim.y + ty;
    unsigned tx = threadIdx.x, ix = blockIdx.x * blockDim.x + tx;

    extern __shared__ real s_a[];

    if (!(iy < M && ix < N)) {
        return;
    } 

    s_a[ty * blockDim.x + tx] = __ldg(&A[iy][ix]);
    __syncthreads();

    unsigned niy = ix - tx + ty;
    unsigned nix = iy - ty + tx;
    B[niy][nix] = s_a[tx * blockDim.x + ty];
}

void transpose_matrix(const real *A, real *B)
{
    const real (*nA)[N] = reinterpret_cast<decltype(nA)>(A);
    real (*nB)[M] = reinterpret_cast<decltype(nB)>(B);

    // block_size应是正方形
    dim3 block_size(32, 32);
    // N是列对应x，M是行对应y
    dim3 grid_size(DIVUP(N, block_size.x), DIVUP(M, block_size.y));
    kernel<<<grid_size, block_size, block_size.y * block_size.x * real_size>>>(nA, nB);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

int main()
{
    launch_gpu();
    return 0;
}
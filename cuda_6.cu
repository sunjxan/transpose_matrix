#include "common.hpp"

// 使用填充共享内存，避免bank冲突

__global__ void kernel(const real (*A)[N], real (*B)[M], size_t sld)
{
    unsigned ty = threadIdx.y, bdy = blockDim.y, iy = blockIdx.y * bdy + ty;
    unsigned tx = threadIdx.x, bdx = blockDim.x, ix = blockIdx.x * bdx + tx;

    extern __shared__ real s_a[];

    if (iy < M && ix < N) {
        s_a[ty * sld + tx] = A[iy][ix];
    } 
    __syncthreads();

    unsigned niy = ix - tx + ty;
    unsigned nix = iy - ty + tx;

    if (niy < N && nix < M) {
        B[niy][nix] = s_a[tx * sld + ty];
    }
}

void transpose_matrix(const real *A, real *B)
{
    const real (*nA)[N] = reinterpret_cast<decltype(nA)>(A);
    real (*nB)[M] = reinterpret_cast<decltype(nB)>(B);

    // block_size应是正方形
    dim3 block_size(32, 32);
    // N是列对应x，M是行对应y
    dim3 grid_size(DIVUP(N, block_size.x), DIVUP(M, block_size.y));
    // 共享内存列数添加pad，sld表示shared memory leading dimension
    size_t pad = 1, sld = block_size.x + pad;
    kernel<<<grid_size, block_size, block_size.y * sld * real_size>>>(nA, nB, sld);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

int main()
{
    launch_gpu();
    return 0;
}
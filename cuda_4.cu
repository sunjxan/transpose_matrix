#include "common.hpp"

// 使用共享内存

__global__ void kernel(const real (*A)[N], real (*B)[M])
{
    unsigned ty = threadIdx.y, bdy = blockDim.y, iy = blockIdx.y * bdy + ty;
    unsigned tx = threadIdx.x, bdx = blockDim.x, ix = blockIdx.x * bdx + tx;

    extern __shared__ real s_a[];

    if (iy < M && ix < N) {
        s_a[ty * bdx + tx] = __ldg(&A[iy][ix]);
    } 
    __syncthreads();

    unsigned v = ty * bdx + tx, nty = v / bdy, ntx = v % bdy;
    unsigned niy = ix - tx + nty;
    unsigned nix = iy - ty + ntx;
    if (niy < N && nix < M) {
        B[niy][nix] = s_a[ntx * bdx + nty];
    }
}

void transpose_matrix(const real *A, real *B)
{
    const real (*nA)[N] = reinterpret_cast<decltype(nA)>(A);
    real (*nB)[M] = reinterpret_cast<decltype(nB)>(B);

    dim3 block_size(15, 17);
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
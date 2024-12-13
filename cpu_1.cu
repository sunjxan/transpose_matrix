#include "common.hpp"

void transpose_matrix(const real *A, real *B)
{
    const real (*nA)[N] = reinterpret_cast<decltype(nA)>(A);
    real (*nB)[M] = reinterpret_cast<decltype(nB)>(B);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            nB[j][i] = nA[i][j];
        }
    }
}

int main()
{
    launch_cpu();
    return 0;
}
#include "common.hpp"

void transpose_matrix(const real *A, real *B)
{
    const real (*nA)[N] = reinterpret_cast<decltype(nA)>(A);
    real (*nB)[M] = reinterpret_cast<decltype(nB)>(B);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            nB[i][j] = nA[j][i];
        }
    }
}

int main()
{
    launch_cpu();
    return 0;
}

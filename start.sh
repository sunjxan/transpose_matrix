# echo "cpu:"
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_1.cu -o cpu_1.out && ./cpu_1.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_1.cu -o cpu_1_dp.out && ./cpu_1_dp.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall cpu_2.cu -o cpu_2.out && ./cpu_2.out
# nvcc -O2 -std=c++17 -Xcompiler -Wall -DUSE_DP cpu_2.cu -o cpu_2_dp.out && ./cpu_2_dp.out
# echo ""
echo "cuda:"
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_1.cu -o cuda_1.out && ./cuda_1.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_1.cu -o cuda_1_dp.out && ./cuda_1_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_2.cu -o cuda_2.out && ./cuda_2.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_2.cu -o cuda_2_dp.out && ./cuda_2_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_3.cu -o cuda_3.out && ./cuda_3.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_3.cu -o cuda_3_dp.out && ./cuda_3_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_4.cu -o cuda_4.out && ./cuda_4.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_4.cu -o cuda_4_dp.out && ./cuda_4_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_5.cu -o cuda_5.out && ./cuda_5.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_5.cu -o cuda_5_dp.out && ./cuda_5_dp.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr cuda_6.cu -o cuda_6.out && ./cuda_6.out
nvcc -O2 -std=c++17 -Xcompiler -Wall --expt-relaxed-constexpr -DUSE_DP cuda_6.cu -o cuda_6_dp.out && ./cuda_6_dp.out

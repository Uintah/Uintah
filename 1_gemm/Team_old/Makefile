gpu: GPU
GPU:
#	nvcc_wrapper -std=c++11 -I/uufs/chpc.utah.edu/common/home/u1009792/temp/test/gemm/poratable/07112018_latest/stk_simd/ -I/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/include -L/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/lib -lkokkos -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -O2 gemm1.cpp -o gpu1

#	nvcc_wrapper -std=c++11 -I/uufs/chpc.utah.edu/common/home/u1009792/temp/test/gemm/poratable/07112018_latest/stk_simd/ -I/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/include -L/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/lib -lkokkos -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -O2 gemm2.cpp -o gpu2

	nvcc_wrapper -std=c++11 -I/uufs/chpc.utah.edu/common/home/u1009792/temp/test/gemm/poratable/07112018_latest/stk_simd/ -I/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/include -L/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/lib -lkokkos -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -O2 gemm3.cpp -o gpu3

#	nvcc_wrapper -std=c++11 -I/uufs/chpc.utah.edu/common/home/u1009792/temp/test/gemm/poratable/07112018_latest/stk_simd/ -I/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/include -L/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/lib -lkokkos -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -O2 gemm4.cpp -o gpu4

	nvcc_wrapper -std=c++11 -I/uufs/chpc.utah.edu/common/home/u1009792/temp/test/gemm/poratable/07112018_latest/stk_simd/ -I/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/include -L/uufs/chpc.utah.edu/common/home/u1009792/install/kingspeak/kokkos/cuda/lib -lkokkos -fopenmp --expt-relaxed-constexpr --expt-extended-lambda -O2 gemm5.cpp -o gpu5

# -arch=sm_37
#
knl: KNL
KNL:
	icpc gemm5.cpp -fopenmp -O3 -g -xMIC-AVX512 -std=c++11 -I/home/sci/damodars/installs/kokkos-knl/build/kokkos/include -I/home/sci/damodars/GEMM/07112018_latest/stk_simd -L/home/sci/damodars/installs/kokkos-knl/build/kokkos/lib -lkokkos -L/home/sci/damodars/installs/hwloc/install/lib -lhwloc -o knl5

	icpc gemm3.cpp -fopenmp -O3 -g -xMIC-AVX512 -std=c++11 -I/home/sci/damodars/installs/kokkos-knl/build/kokkos/include -I/home/sci/damodars/GEMM/07112018_latest/stk_simd -L/home/sci/damodars/installs/kokkos-knl/build/kokkos/lib -lkokkos -L/home/sci/damodars/installs/hwloc/install/lib -lhwloc -o knl3

#!/bin/bash
#SBATCH -t 00:10:00	#expected run time
#SBATCH -N 1 		#number of nodes
#SBATCH -o 1.txt	#output
#SBATCH -e 1.txt	#error
#SBATCH -n 16		#number of mpi tasks
#SBATCH -p soc-gpu-kp  	# Partition on some cluster
#SBATCH -A soc-gpu-kp 	# General CHPC account 
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=32G

# account / partitions names:
# for kingspeak P100: soc-gpu-kp / soc-gpu-kp
# for notchpeak V100: notchpeak-gpu / notchpeak-gpu

export OMP_NUM_THREADS=28
#module load intel/18.1
#module load cuda/9.1


#nvprof --kernels "::TeamTagV2:" --profile-api-trace none --metrics all ../KokkosBatched_Test_Gemm_Cuda.exe -N 16384 -B 5
#nvprof --profile-api-trace none --kernels "::Gemm:" --metrics all ./gpu3 3 16384 2048 32


#./gpu3 3 16384 2048 32
./gpu3 3 16384 256 1
./gpu3 3 16384 256 2
./gpu3 3 16384 256 4
./gpu3 3 16384 256 8
./gpu3 3 16384 256 16
./gpu3 3 16384 256 32

./gpu3 3 16384 512 1
./gpu3 3 16384 512 2
./gpu3 3 16384 512 4
./gpu3 3 16384 512 8
./gpu3 3 16384 512 16
./gpu3 3 16384 512 32

./gpu3 3 16384 1024 1
./gpu3 3 16384 1024 2
./gpu3 3 16384 1024 4
./gpu3 3 16384 1024 8
./gpu3 3 16384 1024 16
./gpu3 3 16384 1024 32

./gpu3 3 16384 2048 1
./gpu3 3 16384 2048 2
./gpu3 3 16384 2048 4
./gpu3 3 16384 2048 8
./gpu3 3 16384 2048 16
./gpu3 3 16384 2048 32



./gpu5 5 16384 256 1
./gpu5 5 16384 256 2
./gpu5 5 16384 256 4
./gpu5 5 16384 256 8
./gpu5 5 16384 256 16
./gpu5 5 16384 256 32

./gpu5 5 16384 512 1
./gpu5 5 16384 512 2
./gpu5 5 16384 512 4
./gpu5 5 16384 512 8
./gpu5 5 16384 512 16
./gpu5 5 16384 512 32

./gpu5 5 16384 1024 1
./gpu5 5 16384 1024 2
./gpu5 5 16384 1024 4
./gpu5 5 16384 1024 8
./gpu5 5 16384 1024 16
./gpu5 5 16384 1024 32

./gpu5 5 16384 2048 1
./gpu5 5 16384 2048 2
./gpu5 5 16384 2048 4
./gpu5 5 16384 2048 8
./gpu5 5 16384 2048 16
./gpu5 5 16384 2048 32


./gpu5 10 16384 256 1
./gpu5 10 16384 256 2
./gpu5 10 16384 256 4
./gpu5 10 16384 256 8
./gpu5 10 16384 256 16
./gpu5 10 16384 256 32

./gpu5 10 16384 512 1
./gpu5 10 16384 512 2
./gpu5 10 16384 512 4
./gpu5 10 16384 512 8
./gpu5 10 16384 512 16
./gpu5 10 16384 512 32

./gpu5 10 16384 1024 1
./gpu5 10 16384 1024 2
./gpu5 10 16384 1024 4
./gpu5 10 16384 1024 8
./gpu5 10 16384 1024 16
./gpu5 10 16384 1024 32

./gpu5 10 16384 2048 1
./gpu5 10 16384 2048 2
./gpu5 10 16384 2048 4
./gpu5 10 16384 2048 8
./gpu5 10 16384 2048 16
./gpu5 10 16384 2048 32

./gpu3 15 16384 256 1
./gpu3 15 16384 256 2
./gpu3 15 16384 256 4
./gpu3 15 16384 256 8
./gpu3 15 16384 256 16
./gpu3 15 16384 256 32

./gpu3 15 16384 512 1
./gpu3 15 16384 512 2
./gpu3 15 16384 512 4
./gpu3 15 16384 512 8
./gpu3 15 16384 512 16
./gpu3 15 16384 512 32

./gpu3 15 16384 1024 1
./gpu3 15 16384 1024 2
./gpu3 15 16384 1024 4
./gpu3 15 16384 1024 8
./gpu3 15 16384 1024 16
./gpu3 15 16384 1024 32

./gpu3 15 16384 2048 1
./gpu3 15 16384 2048 2
./gpu3 15 16384 2048 4
./gpu3 15 16384 2048 8
./gpu3 15 16384 2048 16
./gpu3 15 16384 2048 32


#!/bin/bash
#SBATCH -t 00:30:00	#expected run time
#SBATCH -N 1 		#number of nodes
#SBATCH -o 1.txt	#output
#SBATCH -e 1.txt	#error
#SBATCH -n 1		#number of mpi tasks
#SBATCH -p notchpeak-gpu  	# Partition on some cluster
#SBATCH -A notchpeak-gpu 	# General CHPC account 
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G

# account / partitions names:
# for kingspeak P100: soc-gpu-kp / soc-gpu-kp
# for notchpeak V100: notchpeak-gpu / notchpeak-gpu


module load intel/18.1
module load cuda/9.1

export OMP_NUM_THREADS=32


#./sparse_matvec ./Hamrle1.rb 2048 16 16
#./sparse_matvec ./Trec4.rb 2048 16 16
#./sparse_matvec ./pwtk.rb 2048 16 16

#./sparse_matvec ./pwtk.rb 512 1 256
#./sparse_matvec ./pwtk.rb 512 16 16

./sparse_matvec ./pwtk.rb

#nvprof --profile-api-trace none --kernels "::_spmv___:" -f -o 1.nvvp --source-level-analysis pc_sampling ./sparse_matvec ./pwtk.rb 2048 4

#no bank conflicts
#nvprof --profile-api-trace none --kernels "::_shared_mem_test__:" --events shared_ld_bank_conflict,shared_st_bank_conflict ./sparse_matvec ./pwtk.rb 1 1 32
#nvprof --profile-api-trace none --kernels "::_shared_mem_test__:" --events shared_ld_bank_conflict,shared_st_bank_conflict ./sparse_matvec ./pwtk.rb 1 2 32



#nvprof --profile-api-trace none --kernels "::spmv___:" --events shared_ld_bank_conflict,shared_st_bank_conflict ./sparse_matvec ./pwtk.rb 8192 4 64
#nvprof --profile-api-trace none --kernels "::_shared_mem_test__:" --events shared_ld_bank_conflict,shared_st_bank_conflict ./sparse_matvec ./pwtk.rb 1 4 16
#nvprof --profile-api-trace none --kernels "::_shared_mem_test__:" --events shared_ld_bank_conflict,shared_st_bank_conflict ./sparse_matvec ./pwtk.rb 1 4 8



#nvprof --profile-api-trace none --kernels "::_shared_mem_test__:" --events shared_ld_bank_conflict,shared_st_bank_conflict ./sparse_matvec ./pwtk.rb 1 8 32


#./sparse_matvec ./pwtk.rb 2048 4 64


#fastest: <Team> 4 64 where <Team> in 512, 1024, 2048, 4096, 8192, 16384.. same is cublas +/- 10%

#mpirun -np $SLURM_NTASKS nvprof --profile-api-trace none --kernels "::___csc_spmv___:" -f -o register_test_ilp_in_built.nvvp --source-level-analysis pc_sampling 

if [ 0 -eq 1 ]
then

./sparse_matvec ./pwtk.rb 512 4 8
./sparse_matvec ./pwtk.rb 512 4 16
./sparse_matvec ./pwtk.rb 512 4 32
./sparse_matvec ./pwtk.rb 512 4 64
./sparse_matvec ./pwtk.rb 512 4 128
./sparse_matvec ./pwtk.rb 512 4 256

./sparse_matvec ./pwtk.rb 512 8 4
./sparse_matvec ./pwtk.rb 512 8 8
./sparse_matvec ./pwtk.rb 512 8 16
./sparse_matvec ./pwtk.rb 512 8 32
./sparse_matvec ./pwtk.rb 512 8 64
./sparse_matvec ./pwtk.rb 512 8 128

./sparse_matvec ./pwtk.rb 512 16 4 
./sparse_matvec ./pwtk.rb 512 16 8
./sparse_matvec ./pwtk.rb 512 16 16
./sparse_matvec ./pwtk.rb 512 16 32
./sparse_matvec ./pwtk.rb 512 16 64



./sparse_matvec ./pwtk.rb 1024 4 8
./sparse_matvec ./pwtk.rb 1024 4 16
./sparse_matvec ./pwtk.rb 1024 4 32
./sparse_matvec ./pwtk.rb 1024 4 64
./sparse_matvec ./pwtk.rb 1024 4 128
./sparse_matvec ./pwtk.rb 1024 4 256

./sparse_matvec ./pwtk.rb 1024 8 4
./sparse_matvec ./pwtk.rb 1024 8 8
./sparse_matvec ./pwtk.rb 1024 8 16
./sparse_matvec ./pwtk.rb 1024 8 32
./sparse_matvec ./pwtk.rb 1024 8 64
./sparse_matvec ./pwtk.rb 1024 8 128

./sparse_matvec ./pwtk.rb 1024 16 4 
./sparse_matvec ./pwtk.rb 1024 16 8
./sparse_matvec ./pwtk.rb 1024 16 16
./sparse_matvec ./pwtk.rb 1024 16 32
./sparse_matvec ./pwtk.rb 1024 16 64



./sparse_matvec ./pwtk.rb 2048 4 8
./sparse_matvec ./pwtk.rb 2048 4 16
./sparse_matvec ./pwtk.rb 2048 4 32
./sparse_matvec ./pwtk.rb 2048 4 64
./sparse_matvec ./pwtk.rb 2048 4 128
./sparse_matvec ./pwtk.rb 2048 4 256

./sparse_matvec ./pwtk.rb 2048 8 4
./sparse_matvec ./pwtk.rb 2048 8 8
./sparse_matvec ./pwtk.rb 2048 8 16
./sparse_matvec ./pwtk.rb 2048 8 32
./sparse_matvec ./pwtk.rb 2048 8 64
./sparse_matvec ./pwtk.rb 2048 8 128

./sparse_matvec ./pwtk.rb 2048 16 4 
./sparse_matvec ./pwtk.rb 2048 16 8
./sparse_matvec ./pwtk.rb 2048 16 16
./sparse_matvec ./pwtk.rb 2048 16 32
./sparse_matvec ./pwtk.rb 2048 16 64



./sparse_matvec ./pwtk.rb 4096 4 8
./sparse_matvec ./pwtk.rb 4096 4 16
./sparse_matvec ./pwtk.rb 4096 4 32
./sparse_matvec ./pwtk.rb 4096 4 64
./sparse_matvec ./pwtk.rb 4096 4 128
./sparse_matvec ./pwtk.rb 4096 4 256

./sparse_matvec ./pwtk.rb 4096 8 4
./sparse_matvec ./pwtk.rb 4096 8 8
./sparse_matvec ./pwtk.rb 4096 8 16
./sparse_matvec ./pwtk.rb 4096 8 32
./sparse_matvec ./pwtk.rb 4096 8 64
./sparse_matvec ./pwtk.rb 4096 8 128

./sparse_matvec ./pwtk.rb 4096 16 4 
./sparse_matvec ./pwtk.rb 4096 16 8
./sparse_matvec ./pwtk.rb 4096 16 16
./sparse_matvec ./pwtk.rb 4096 16 32
./sparse_matvec ./pwtk.rb 4096 16 64



 
fi

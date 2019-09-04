#!/bin/bash

baseline_num_blocks=1
baseline_num_threads=256
baseline_num_streams=1

filename=char-ox-16p-patchfixed

#run the baseline

/usr/local/cuda-8.0/bin/nvprof --profile-api-trace none -f -o ${filename}_${baseline_num_blocks}sm_${baseline_num_threads}thr_${baseline_num_streams}str.nvvp /home/brad/opt/uintah/branch-gpu-kokkos/StandAlone/sus -nthreads 16 -gpu -cuda_blocks_per_loop ${baseline_num_blocks} -cuda_threads_per_block ${baseline_num_threads} ${filename}.ups | tee ${filename}_${baseline_num_blocks}sm_${baseline_num_threads}thr_${baseline_num_streams}str-output.txt

declare -a num_blocks=(2 3 4 8 16 24 32 48)
for item in "${num_blocks[@]}"
do
  filedesc=${filename}_${item}sm_${baseline_num_threads}thr_${baseline_num_streams}str
  /usr/local/cuda-8.0/bin/nvprof --profile-api-trace none -f -o ${filedesc}.nvvp /home/brad/opt/uintah/branch-gpu-kokkos/StandAlone/sus -nthreads 16 -gpu -cuda_blocks_per_loop ${item} -cuda_threads_per_block ${baseline_num_threads} ${filename}.ups | tee ${filedesc}-output.txt
done

declare -a num_threads=(128 160 192 224 288)
#declare -a num_threads=(160)
for item in "${num_threads[@]}"
do
  filedesc=${filename}_${baseline_num_blocks}sm_${item}thr_${baseline_num_streams}str
  /usr/local/cuda-8.0/bin/nvprof --profile-api-trace none -f -o ${filedesc}.nvvp /home/brad/opt/uintah/branch-gpu-kokkos/StandAlone/sus -nthreads 16 -gpu -cuda_blocks_per_loop ${baseline_num_blocks} -cuda_threads_per_block ${item} ${filename}.ups | tee ${filedesc}-output.txt
done




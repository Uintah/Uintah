#!/bin/bash

#PBS -V
#PBS -j oe
#PBS -M t.harman@utah.edu  KEITA CHANGE ME
#PBS -m abe


export SCI_DEBUG='ProgressiveWarning:-,ComponentTimings:+'

#export MPICH_UNEX_BUFFER_SIZE=120M
#export MPICH_PTL_OTHER_EVENTS=4096
export MPICH_VERSION_DISPLAY=1
export MPICH_ENV_DISPLAY=1

cd $SCRIPT_DIR
cd output

FILE="out.$JOB.$nodes.$size"
EXE=../../sus        KEITA CHANGE ME

echo "---------------------------------"
date
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "ups=$nodes.ups"
echo "procs=$procs"
echo "size=$size"
echo "threads=$threads"
echo "way=$WAY"
echo "nodes=$nodes"
echo "pwd=`pwd`"
echo "---------------------------------"
    
pwd
if [ "$threads" -eq "1" ]; then
  echo "aprun -n $procs  $EXE -mpi -do_not_validate  ../tmp/$ups &> $FILE"
  aprun -n $procs $EXE -mpi -do_not_validate  ../tmp/$ups &> $FILE
else 
  export MPICH_MAX_THREAD_SAFETY=multiple
  echo "aprun -n $procs -N $WAY -d $threads $EXE -mpi -do_not_validate -nthreads $threads ../tmp/$ups &> $FILE"
  aprun -n $procs -N $WAY -d $threads $EXE -mpi -do_not_validate -nthreads $threads ../tmp/$ups &> $FILE
fi



#!/bin/bash

PROJECT="cmb109"

NODES="512 1024 2048 4096 8192 16384"
UPS="512.ups 1024.ups 2048.ups 4096.ups 8192.ups 16384.ups"
JOB="rmcrt.bm1.do.large"

WAY="1"
THREADS="16"

num_timesteps="10"
course_resolution="[128,128,128]"
course_patches="[8,8,8]"
fine_resolution="[512,512,512]"
random_seed="true"
num_rays="100"
halo_region="[4,4,4]"

ups="512.ups"
fine_patches="[32,16,16]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="1024.ups"
fine_patches="[32,32,16]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="2048.ups"
fine_patches="[32,32,32]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="4096.ups"
fine_patches="[64,32,32]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="8192.ups"
fine_patches="[64,64,32]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="16384.ups"
fine_patches="[64,64,64]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups


SCRIPT_DIR=`pwd`
export SCRIPT_DIR

rm -f output/*

for nodes in $NODES; do
    
    size=`expr $nodes \* 16`
    ups=$nodes.ups
    
    if [ $nodes -lt 2048 ]; then
      TIME=02:00:00
    elif [ $nodes -lt 4096 ]; then
      TIME=01:00:00
    elif [ $nodes -lt 8192 ]; then
      TIME=01:30:00
    else
      TIME=01:00:00
    fi

    if [ $THREADS -eq 1 ]; then
      procs=$size
      mode=$MODE
      threads=1
    else
      procs=$nodes
      mode=1
      threads=$THREADS
    fi 

    export JOB
    export ups
    export size
    export procs
    export WAY
    export threads
    export nodes
      
    echo "qsub -A $PROJECT -N $JOB.$threads.$nodes.$size -l walltime=$TIME,nodes=$nodes,gres=atlas1%atlas2 ../runsus.sh"
    qsub -A $PROJECT -N $JOB.$threads.$nodes.$size -l walltime=$TIME,nodes=$nodes,gres=atlas1%atlas2 ../runsus.sh
    
done


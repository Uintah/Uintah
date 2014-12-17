#!/bin/bash

PROJECT="cmb109"

NODES="8 16 32 64 128 256"
UPS="8.ups 16.ups 32.ups 64.ups 128.ups 256.ups"
JOB="rmcrt.bm1.do.small"

WAY="1"
THREADS="16"

num_timesteps="10"
course_resolution="[32,32,32]"
course_patches="[2,2,2]"
fine_resolution="[128,128,128]"
random_seed="true"
num_rays="100"
halo_region="[4,4,4]"

ups="8.ups"
fine_patches="[8,4,4]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="16.ups"
fine_patches="[8,8,4]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="32.ups"
fine_patches="[8,8,8]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="64.ups"
fine_patches="[16,8,8]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="128.ups"
fine_patches="[16,16,8]"
cp ../skel.rmcrt.bm1.DO.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<course_resolution>>/$course_resolution/" tmp/$ups
perl -pi -w -e "s/<<course_patches>>/$course_patches/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<random_seed>>/$random_seed/" tmp/$ups
perl -pi -w -e "s/<<num_rays>>/$num_rays/" tmp/$ups
perl -pi -w -e "s/<<halo_region>>/$halo_region/" tmp/$ups

ups="256.ups"
fine_patches="[16,16,16]"
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
    
    if [ $nodes -lt 128 ]; then
      TIME=02:00:00
    elif [ $nodes -lt 256 ]; then
      TIME=01:00:00
    elif [ $nodes -lt 512 ]; then
      TIME=02:00:00
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


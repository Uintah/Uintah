#!/bin/bash

PROJECT="cmb109"    KEITA CHANGE ME

#NODES="2"
NODES="2 4 8 16 32 64 128"
UPS="2.ups 4.ups 8.ups 16.ups 32.ups 64.ups 128.ups"
JOB="small"

WAY="1"
THREADS="16"
num_timesteps="25"
fine_resolution="[256,256,256]"
fine_patches="[16,16,16]"
scheduler='"Unified"'

ups="2.ups"
fine_patches="[4,4,2]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="4.ups"
fine_patches="[4,4,4]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="8.ups"
fine_patches="[8,4,4]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="16.ups"
fine_patches="[8,8,4]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="32.ups"
fine_patches="[8,8,8]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="64.ups"
fine_patches="[16,8,8]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="128.ups"
fine_patches="[16,16,8]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="256.ups"
fine_patches="[16,16,16]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

SCRIPT_DIR=`pwd`
export SCRIPT_DIR

#rm -f output/*

for nodes in $NODES; do
    
    size=`expr $nodes \* 16`
    ups=$nodes.ups
    
    if [ $nodes -lt 4 ]; then
      TIME=02:00:00
    elif [ $nodes -lt 16 ]; then
      TIME=01:30:00
    elif [ $nodes -lt 64 ]; then
      TIME=01:00:00
    else
      TIME=00:30:00
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

    FILE=$JOB.$threads.$nodes.$size.out
    if [ -f $FILE ]; then
      rm -f output/$FILE
    fi
      
    echo "qsub -A $PROJECT -N $JOB.$threads.$nodes.$size -l walltime=$TIME,nodes=$nodes,gres=atlas1%atlas2 ../runsus.sh"
    qsub -A $PROJECT -N $JOB.$threads.$nodes.$size -l walltime=$TIME,nodes=$nodes,gres=atlas1%atlas2 ../runsus.sh
    
done


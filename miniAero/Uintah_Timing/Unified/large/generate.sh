#!/bin/bash

PROJECT="cmb109"        KEITA CHANGE ME

NODES="128 256 512 1024 2048 4096 8192"
UPS="128.ups 256.ups 512.ups 1024.ups 2048.ups 4096.ups 8192.ups"
JOB="large"

WAY="1"
THREADS="16"
num_timesteps="25"
fine_resolution="[1024,1024,1024]"
scheduler='"Unified"'

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

ups="512.ups"
fine_patches="[32,16,16]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="1024.ups"
fine_patches="[32,32,16]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="2048.ups"
fine_patches="[32,32,32]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="4096.ups"
fine_patches="[64,32,32]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="8192.ups"
fine_patches="[64,64,32]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

ups="16384.ups"
fine_patches="[64,64,64]"
cp ../skel.riemann3D.ups tmp/$ups
perl -pi -w -e "s/<<num_timesteps>>/$num_timesteps/" tmp/$ups
perl -pi -w -e "s/<<fine_resolution>>/$fine_resolution/" tmp/$ups
perl -pi -w -e "s/<<fine_patches>>/$fine_patches/" tmp/$ups
perl -pi -w -e "s/<<scheduler>>/$scheduler/" tmp/$ups

SCRIPT_DIR=`pwd`
export SCRIPT_DIR

for nodes in $NODES; do
    
    size=`expr $nodes \* 16`
    ups=$nodes.ups
    
    if [ $nodes -lt 256 ]; then
      TIME=02:00:00
    elif [ $nodes -lt 1024 ]; then
      TIME=01:30:00
    elif [ $nodes -lt 4096 ]; then
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

    FILE=$JOB.$threads.$nodes.$size.out
    if [ -f $FILE ]; then
      echo
      echo "removing $FILE"
      echo
      rm -f output/$FILE
    fi
      
    echo "qsub -A $PROJECT -N $JOB.$threads.$nodes.$size -l walltime=$TIME,nodes=$nodes,gres=atlas1%atlas2 ../runsus.sh"
    qsub -A $PROJECT -N $JOB.$threads.$nodes.$size -l walltime=$TIME,nodes=$nodes,gres=atlas1%atlas2 ../runsus.sh
    
done


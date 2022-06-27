#!/bin/bash

#
# The MIT License
#
# Copyright (c) 1997-2021 The University of Utah
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

#______________________________________________________________________
#  This script creates a series of movie frames in parallel using the slurm batch scheduler.
#  It assumes that the user has created a session file using the server client
#  option in VisIt and has copied it over to the remote machine.
#  See:
#       https://www.visitusers.org/index.php?title=Making_Movies
#  for additional details
#  
#   Warning: This script works without issues most of the time.  Occasionally,
#   a job will not produce movie frames, while many other jobs work correctly. 
#   
#______________________________________________________________________
#             main
#
main()
{
  frameLo=835              # timestep to start making movie
  frameHi=840              # timestep to stop
  increment=20             # number of frames per node

  #     USER DEFINED VARIABLES
  declare -g visIt="/uufs/chpc.utah.edu/common/home/harman-group3/harman/VisIt/bin/visit"
  declare -g version="3.2.2"
  declare -g sessionFile="/uufs/chpc.utah.edu/common/home/harman-group3/harman/RZero/JPMC_confRoom/v01.0.1/volVis/volVis.session"
  declare -g outputDir="/uufs/chpc.utah.edu/common/home/harman-group3/harman/RZero/JPMC_confRoom/v01.0.1/volVis"
  declare -g sourceUda="/uufs/chpc.utah.edu/common/home/harman-group3/harman/RZero/JPMC_confRoom/v01.0.1/masterUda/index.xml"
  declare -g nNodes=1
  declare -g nProcs=20
  declare -g time=4:00:00
  declare -g geometry=1952x780
  declare -g partition=smithp-ash
  declare -g bank=smithp-ash-cs
  declare -g desc=v01
  declare -g start=0

  if [[ (! -f "$sourceUda") || (! -f "$sessionFile") ]]; then
    echo "ERROR: the sourceUda or session file was not found"
    exit
  fi

  #__________________________________
  # Loop over the frames using the increment
   for (( start=frameLo; start<frameHi; start+=increment )); do

    declare -g end=$((start+increment))
    declare -g jobName="movie-$desc-$start-$end"

    sbatch_header

    VisIt_cmd

    sbatch batch.slrm
    echo " working $start $end $frameHi"

    sleep 1
  done
}


#______________________________________________________________________
#       Function that writes out the SBATCH commands to the top level script
sbatch_header()
{
cat << EOF > batch.slrm
#!/bin/bash
#SBATCH --nodes $nNodes
#SBATCH --ntasks $nProcs
#SBATCH --partition $partition
#SBATCH --account $bank
#SBATCH --time $time
#SBATCH --job-name $jobName
#SBATCH --output "out.$jobName"
#SBATCH --exclude=ash253
EOF
}


#______________________________________________________________________
#       Function that concatenates to the batch.slrm script the VisIt
#       command.
#  Be careful!  No spaces after "\"
VisIt_cmd()
{
cat << EOF >> batch.slrm
  "$visIt"  \
  -v $version -nowin\\
  -movie -start $start -end $end -frame $start \\
  -format png \
  -geometry "$geometry"\
  -fps 10 \\
  -ignoresessionengines \\
  -par \
  -l  sbatch/mpirun\
  -p  $partition \
  -b  $bank \
  -n  $jobName \
  -nn $nNodes \
  -np $nProcs\\
  -source $sourceUda \\
  -output $outputDir/movie.  \\
  -sessionfile $sessionFile
EOF
}

#______________________________________________________________________
#______________________________________________________________________
main "$@"
exit


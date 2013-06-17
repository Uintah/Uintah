#! /bin/sh
###############################################################################
# validateInputs
#   Validates that all the files in ICE/MPMICE/UCF/Models run with current
#   code and prints out failed messages. Will either check a specific directory
#   or all directories within either current or <sourceDirectory>.
###############################################################################
# usage:
#    validateInputs <optional: sourceDirectory> <optional: specificDir>
###############################################################################

# Set Constants 
flags=0
checkDirectory="grr"
defaultDir="../StandAlone/inputs"
usage0="Run in '$defaultDir'"
usage1="This utility checks each *ups file in the directories contained in the "
usage2="current directory or optionally the specified directory.  It prints out"
usage3=" the exit status or exception if one is thrown."
usage4="Usage: validateInputs <optional: specific directory to search>"


# Logic for base directory
if [ $# -gt 0 ]
then
  if [ "$1" = "-h" ]
  then
    echo
    echo $usage0
    echo $usage1
    echo $usage2
    echo $usage3
    echo $usage4
    echo
    exit 0
  fi
  defaultDir=$1
fi

# Logic for checking only one directory
if [ $# -gt 1 ]
then
  checkDirectory=$2
  flags=$#
fi


# Move to directory and start checking
cd $defaultDir
# Start First For Loop
for directorys in `find * -type d -prune` 
do
  # Check if only perform on one directory
  if [ $flags -gt 1 ]
  then
    if [ $checkDirectory != $directorys ]
    then
      continue
    fi
  fi

  cd $directorys
  echo "Checking $directorys Files"
  echo "--------------------------"

  # Start Second For Loop
  for file in `ls .`
  do
    extens=$(basename $file |awk -F. '{print $NF}')
    if [ "$extens" = 'ups' ] 
    then
      # Copy file such that it is not modified
      cat $file > upsfile.ups
      echo "--------------------------------------------------------------"
      echo "Checking $file......."
      # Add one timestep option
      sed -i /\<Time\>/a\ "<max_Timesteps> 1 </max_Timesteps>" upsfile.ups

      # Run file
      echo "Running $file"
      mpirun.openmpi -np 1 sus upsfile.ups >& outfile

      # Print results
      grep Sus outfile
      grep Thread outfile
      grep -A 10 Caught outfile 
      echo

      # Clean up before next iteration
      rm -rf upsfile.ups outfile *uda*
    fi

  # End first for loop
  done
  cd ..

# End second for loop
done



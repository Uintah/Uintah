#! /bin/sh

#
# parallel_extract.sh
#
# [MAKE SURE TO SET THE VARIABLES BELOW BEFORE RUNNING THE SCRIPT.  See
#  'SET THE FOLLOWING VARIABLES' below.]
#
# This script runs uda2nrrd to generate NRRDs using 
# multiple processors at the same time.
#
# Note, this script currently only handles volume variables (not particles),
# though it would be trivial to update it to handle particles.  Exercise to the
# reader: handle the usual uda2nrrd command line args via command line
# to this script... see following paragraph also:
#
# The next step probably would be to make all the following variables
# into command line args... but for now that is left as an exercise to the
# reader. ;)
#
# CAVEATS: 
#   - If the NRRD files already exist, uda2nrrd will ask you if you wish to
#       overwrite them... however, since you are running in 'batch' mode,
#       you can't respond... so make sure not to have NRRDs already created.
#
#   - Note, you should get a good speed up using this script... however, if
#       for some reason, the uda2nrrd doesn't run in about the same time
#       for all timesteps (in a given batch), then the one uda2nrrd that 
#       takes a long time would block the script from moving on to the
#       next batch of jobs, thus slowing everything down.  In practice, this
#       shouldn't be a problem.  Note, I don't see a good way to fix this
#       as I don't see a way to wait for multiple jobs to finish and 
#       to continue with new jobs for any of the jobs that finish... I believe
#       you can only wait for one job at a time...

####################################################################
# Make sure to SET THE FOLLOWING VARIABLES:
#

# The UDA you wish to extract data from:

UDA=~/SVN/SCIRun/inferno32opt/StandAlone/vab_newtable_2motors.uda

# Location of uda2nrrd executable:

UDA2NRRD=~/SVN/SCIRun/inferno32opt/StandAlone/tools/uda2nrrd/uda2nrrd

# Index of first timestep to extract:

number=0

# Index of last timestep to extract:

end=100

# Number of jobs to run at the same time:

batch_size=8

# Name of variable to extract (Note, it can't be a praticle var...):

VARIABLE=tempIN

####################################################################
#
# Actual script begins here:
#


####################################################################
#
# Do some input validation:

if ! test -d $UDA; then
  echo
  echo "Error: $UDA does not exist!  Goodbye."
  echo
  exit 1
fi

if ! test -f $UDA2NRRD; then
  echo
  echo "Error: $UDA2NRRD does not exist!  Goodbye."
  echo
  exit 1
fi

if test $batch_size -gt 16; then
  echo
  echo "Error:  I think you accidentally specified too many processors ($batch_size) to use...  Goodbye."
  echo
  exit 1
fi

batch_end=`echo $number + $batch_size | bc`

####################################################################
#
# do_group()
#
#    Kicks off a batch of processes
#

function do_group {
  pids=
  while test $number -lt $batch_end -a $number -le $end; do

    $UDA2NRRD \
            \
            -v $VARIABLE -m 2  \
            \
            -tstep $number     \
            \
            -uda $UDA &

    # Record the PID of the uda2nrrd background process so we can wait for it later...
    pid=$!

    echo "Extracting timestep $number.  (PID is: $pid)"

    pids="$pid $pids"

    number="`echo $number + 1 | bc`"
  done
  batch_end=`echo $number + $batch_size | bc`
}

####################################################################
#
# main()
#
#    Calls the do_group() function to run a batch of processes, then waits
# for all those processes to finish... loop.
#

done="false"

while test $done != "true" -a $number -le $end; do
  do_group
  
  for pid in $pids; do
    wait $pid
    result=$?
    if test "$result" -ne 0; then
      echo
      echo "uda2nrrd (pid: $pid) failed (return code: $result)..."
      echo "Exiting after all current extraction processes finish..."
      echo
      done="true"
    fi
  done

  if test $done = "true"; then
    echo
    echo "ERROR: All currently running extractions have now completed... At least one"
    echo "       failed..."
    echo
  fi
done




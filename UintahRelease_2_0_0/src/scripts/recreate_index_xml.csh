#! /bin/tcsh

#
# Author: J. Davison de St. Germain
# Date:   8/4/2016
#
# This script is used to recreate the <timesteps> section of an index.xml for a UDA
# when the index.xml is accidentally deleted.  This script is most useful when you
# have an index.xml from another (very similar) UDA that you can use as the template
# and then just replace the <timesteps> section.
#
# This script does not create the entire index.xml file.  In order to do so, you would
# need the output from the original 'sus' run, in addition to the data in some of the
# UDA files that this script uses.
#

set UDA=/scratch/ash/lustre/spinti/handoff/ifrf_handoff.uda.008

echo
echo "Creating <timesteps> section for new index.xml for uda: $UDA"
echo

cd $UDA

set tsteps=`ls -1trd t* | grep -v totalKineticEnergy`

echo "  <timesteps>"

foreach dir ( $tsteps )

   set num=`echo $dir | cut -f2 -d"t"`
   set time=`grep currentTime $dir/timestep.xml | cut -f2 -d">" | cut -f1 -d"<"`
   set olddt=`grep oldDelt $dir/timestep.xml | cut -f2 -d">" | cut -f1 -d"<"`

   echo "    <timestep href="'"'$dir/timestep.xml'" 'time='"'$time'"' oldDelt='"'$olddt'">'$num"</timestep>"
end

echo "  </timesteps>"

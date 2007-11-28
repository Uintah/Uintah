#!/bin/bash

#
# sphere2nrrd.sh 
#
#   Script that takes a .raw file (of particle information) and
#   creates a .nhdr file (Nrrd header) for it.
#
#   Author: James Bigler, J. Davison de St. Germain
#

# $1 - raw file name
# $2 - number of variables

if [ $# != 2 ]
then
  echo "Usage: $0 filename.raw num_variables"
  echo ""
  echo "Creates a nrrd header for the given file."
  exit 1
fi

unu=`which unu 2> /dev/null`
if [ -z $unu ]
then
  echo ""
  echo "ERROR: Can't find the 'unu' program...  Please make sure it is in your path.  Goodbye."
  echo
  exit 1
fi

if [ ! -f $1 ]
then
  echo ""
  echo "ERROR: Can't open raw datafile '$1'.  Goodbye."
  echo
  exit 1
fi


num_parts=`ls -l $1 | awk '{print $5}' | xargs -i\{} echo "scale=0;"\{}/$2/4 | bc`

nrrd_header=$1.nhdr

echo unu make -h -i $1 -o $nrrd_header -t float -s $2 $num_parts
unu make -h -i $1 -o $nrrd_header -t float -s $2 $num_parts



#! /bin/bash
#
# The MIT License
#
# Copyright (c) 1997-2023 The University of Utah
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
#/
#______________________________________________________________________

usage()
{
  echo "______________________________________________________________________"
  echo
  echo "DESCRIPTION"
  echo " makeCombinedIndex.sh creates a list of timesteps from multiple udas"
  echo " and generates a single index.xml file that you can point to."
  echo " "
  echo "Usage:"
  echo "         makeCombinedIndex.sh [options]  <udas>"
  echo "Options:"
  echo "    -h, --help          display usage"
  echo "    -f, --fullpath      output the full path to the uda the default is the relative path"
  echo "______________________________________________________________________"
  exit
}

#______________________________________________________________________
#   Assumption:  script is run at same directory level as masterUda
#______________________________________________________________________
main()
{
  fullPath="false"

  #__________________________________
  # parse arguments
  options=$( getopt --name "makeCombinedIndex.sh" --options="h,f"  --longoptions=help,fullpath -- "$@" )

  if [[ $? -ne 0 || $# -eq 0 ]] ; then
    echo "Incorrect option provided"
    usage
    exit 1
  fi

  # set is to preserve white spaces and punctuation
  eval set -- "$options"

  while true ; do
    case "$1" in
      -f|--fullpath)
        fullPath="true"
        ;;
      -h|--help)
        usage
        ;;
      --)
        shift
        break
        ;;
    esac
    shift
  done

  udas=($@)
  #__________________________________
  # bulletproofing
  for dir in "${udas[@]}"; do
    if test ! -d "$dir"; then
      echo "ERROR: '$dir' is not a directory!  Goodbye."
      exit
    fi
  done

  preTimestep=-1

  #__________________________________
  # loop over udas
  for uda in "${udas[@]}"; do

    echo "Processing $uda">&2

    timesteps=( $( grep "timestep href=" "$uda/index.xml" | awk -F'[=/"]' '{ print $3 }') )

    for timestep in "${timesteps[@]}"; do

      tsFilePath="$uda/$timestep/timestep.xml"

      # does timestep file exist
      if test -f "$tsFilePath"; then

        tsNum=$( echo $timestep | cut -d"t" -f2 )

        if test "$tsNum" -le "$preTimestep"; then
          echo "    WARNING: $uda: $timestep has already been added to masterUda/index.xml.  Ignoring it." >&2
        else

          #  define the full path or relative path
          if test "$fullPath" == "true"; then
            udaPath=$( realpath "$uda" )
          else
            udaPath=$( realpath --relative-to=masterUda "$uda" )
          fi

          tsFilePath="$udaPath/$timestep/timestep.xml"

          timeline=$(grep "timestep href=" "$uda/index.xml" | grep $timestep )
          time=$(    echo $timeline | cut -f3 -d" " | cut -f2 -d"=" )
          oldDelt=$( echo $timeline | cut -f4 -d" " | cut -f2 -d"=" | cut -f1 -d">" )
          timeNum=$( echo $timeline | cut -f4 -d" " | cut -f2 -d">" | cut -f1 -d"<" )

          echo "    "\<timestep href=\"$tsFilePath\" time=$time oldDelt=$oldDelt\>$timeNum\</timestep\> >&2
          echo "    "\<timestep href=\"$tsFilePath\" time=$time oldDelt=$oldDelt\>$timeNum\</timestep\>
          preTimestep=$tsNum
        fi
      else
        echo "ERROR $tsFilePath does not exist" >&2
      fi
    done
  done
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"

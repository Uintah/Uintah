#!/bin/bash
# The MIT License
#
# Copyright (c) 1997-2024 The University of Utah
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

#______________________________________________________________________
#  monitorSus
#  This script searches the output file every N minutes and 
#  if the timestep number has not increased then the job ID is killed.
#______________________________________________________________________

helpersPath=$( dirname "${BASH_SOURCE[0]}")
source "${helpersPath}"/bashFunctions

#______________________________________________________________________

usage()
{
  echo ""
  echo " Usage: $0 "
  echo " This script searches the output file every N minutes for"
  echo " an increase in the timestep number.  If the timestep"
  echo " is not increasing the jobID is killed with a scancel command"
  echo ""
  echo " Manditory inputs:"
  echo "   --jid             <job id that will be killed>"
  echo "   --out             <name of the output file to monitor>"
  echo "   --initialInterval <interval in minutes before the first search>"
  echo "   --interval        <interval in minutes between search>"
  exit 1
}



#______________________________________________________________________
#
#______________________________________________________________________
main()
{
  options=$( getopt --name "monitorSus.sh" --options="h" --longoptions=out:,jid:,interval:,initialInterval:,help -- "$@" )

  if [[ $? -ne 0 || $# -ne 8 ]]; then
    echo "Terminating..." >/dev/stderr
    usage
    exit 1
  fi

  eval set -- "$options"

  declare -i jid=-9
  initialInterval=-9
  interval=-9

  while true ; do

    case "$1" in
      --jid)
        shift
        jid="$1"
        ;;
      --out)
        shift
        out="$1"
        ;;
       --initialInterval)
        shift
        initialInterval="$1"
        ;;
       --interval)
        shift
        interval="$1"
        ;;
      -help|--help|-h)
        usage
        ;;
       --)
        shift
        break
        ;;
      *)
        echo -e "\n    ERROR: ($1) Unknown option."
        usage
        exit
        ;;
    esac
    shift
  done
  
  #__________________________________
  #  bulletproofing
  killCmd=$( which scancel )                # slurm

  if [[ $? -ne 0 ]]; then
    killCmd=$( which qdel )                 # pbs
  fi


  if [[ ! -e "$killCmd" ]]; then
    printf "\n ERROR: Could not find the command to kill the job.\n"
    exit
  fi

  if [[ ! -e $out ]]; then
    printf "\n ERROR: Could not find the output (%s)\n", "$out"
    exit
  fi

#__________________________________
  declare -i ts_old=0

  sleepMin=$initialInterval"m"    # first time through the loop

  while true ; do

    sleep "$sleepMin"
    sleepMin=$interval"m"

    # Has the simulation ended
    declare -i hasEnded
    grep --silent "Sus: going down successfully" "$out"
    hasEnded=$?
    
    declare -i ts
    ts=$( grep Timestep "$out" | tr -d '[:punct:]' | awk 'END {print $2}' )
    
    if [[ $ts -eq $ts_old || $hasEnded  -eq 0 ]]; then
      echo "__________________________________"
      date
      echo "  MonitorSus.sh: Now killing job $jid on timestep $ts_old ($killCmd)"
      #$killCmd "$jid"

      exit
    else
      ts_old=$ts
    fi

  done

exit
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"

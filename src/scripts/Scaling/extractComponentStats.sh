#!/bin/bash

#______________________________________________________________________
#  This script, parses the SCI_DEBUG ComponentNodeStats:+, or ComponentStats:+ output, between starting and ending timesteps
#  and computes an average  The only difference between the two SCI_DEBUG tags is the summation in the ComponentNodeStats
#
#                ComponentStats
# Runtime Summary  summary stats for time step 1 at time=1e-09
# Description       Units       Minimum           Rank   Average (2)       Std. Dev.         Maximum           Rank   100*(1-ave/max) '% load imbalance'
# Compilation       [seconds] : 0.000401624     : 1    : 0.000430569     : 4.09344e-05     : 0.000459514     : 0    : 6.29905
# TaskExec          [seconds] : 0.676202        : 0    : 0.676294        : 0.000129922     : 0.676386        : 1    : 0.0135823
# TaskLocalComm     [seconds] : 0.0106768       : 0    : 0.0116087       : 0.00131787      : 0.0125406       : 1    : 7.43085
# TaskWaitCommTime  [seconds] : 0.00267854      : 0    : 0.0028042       : 0.000177716     : 0.00292987      : 1    : 4.28907
# NumberOfTasks     [tasks  ] : 18              : 0    : 18              : 0               : 18              : 0    : 0
# NumberOfPatches   [patches] : 1               : 0    : 1               : 0               : 1               : 0    : 0
# NumberOfCells     [cells  ] : 1.0985e+06      : 0    : 1.0985e+06      : 0               : 1.0985e+06      : 0    : 0
# SCIMemoryUsed     [MBytes ] : 4.05348e+08     : 1    : 4.21714e+08     : 2.31444e+07     : 4.38079e+08     : 0    : 3.73575
# SCIMemoryMaxUsed  [MBytes ] : 4.05348e+08     : 1    : 4.21714e+08     : 2.31444e+07     : 4.38079e+08     : 0    : 3.73575
# MemoryUsed        [MBytes ] : 3.13449e+09     : 1    : 3.18042e+09     : 6.49497e+07     : 3.22635e+09     : 0    : 1.42348
# MemoryResident    [MBytes ] : 4.05348e+08     : 1    : 4.21714e+08     : 2.31444e+07     : 4.38079e+08     : 0    : 3.73575
# Percentage of time spent in overhead : 0.0622986
#
#
#               ComponentNodeStats
# Runtime Node c515-001.stampede3.tacc.utexas.edu summary stats for time step 1 at time=1e-09
# Description       Units       Sum (2)           Minimum           Rank   Average (2)       Std. Dev.         Maximum           Rank   100*(1-ave/max) '% load imbalance'
# Compilation       [seconds] : 0.000861138     : 0.000401624     : 1    : 0.000430569     : 4.09344e-05     : 0.000459514     : 0    : 6.29905
# TaskExec          [seconds] : 1.35259         : 0.676202        : 0    : 0.676294        : 0.000129922     : 0.676386        : 1    : 0.0135823
# TaskLocalComm     [seconds] : 0.0232174       : 0.0106768       : 0    : 0.0116087       : 0.00131787      : 0.0125406       : 1    : 7.43085
# TaskWaitCommTime  [seconds] : 0.00560841      : 0.00267854      : 0    : 0.0028042       : 0.000177716     : 0.00292987      : 1    : 4.28907
# NumberOfTasks     [tasks  ] : 36              : 18              : 0    : 18              : 0               : 18              : 0    : 0
# NumberOfPatches   [patches] : 2               : 1               : 0    : 1               : 0               : 1               : 0    : 0
# NumberOfCells     [cells  ] : 2.197e+06       : 1.0985e+06      : 0    : 1.0985e+06      : 0               : 1.0985e+06      : 0    : 0
# SCIMemoryUsed     [MBytes ] : 8.43428e+08     : 4.05348e+08     : 1    : 4.21714e+08     : 2.31444e+07     : 4.38079e+08     : 0    : 3.73575
# SCIMemoryMaxUsed  [MBytes ] : 8.43428e+08     : 4.05348e+08     : 1    : 4.21714e+08     : 2.31444e+07     : 4.38079e+08     : 0    : 3.73575
# MemoryUsed        [MBytes ] : 6.36084e+09     : 3.13449e+09     : 1    : 3.18042e+09     : 6.49497e+07     : 3.22635e+09     : 0    : 1.42348
# MemoryResident    [MBytes ] : 8.43428e+08     : 4.05348e+08     : 1    : 4.21714e+08     : 2.31444e+07     : 4.38079e+08     : 0    : 3.73575



#______________________________________________________________________
#         Parse the output and clean out everything except
#         the lines assocated with SCI_DEBUG

extract_tag_lines()
{
  local tag=$1

  if [[ "$tag" == "ComponentStats" ]]; then                     #   HARDCODED!
    searchFor="Runtime Summary  summary stats for time step"
  else
    searchFor="Runtime Node "
  fi

  # extract all text between "$tag" and MemoryResident

  sed -n '/'"$searchFor"'/,/'"  MemoryResident"'/p'  "$out_clean"    >& "$out_clean2"

  if [[ "$?" != "0" ]]; then
    echo " There was a problem extracting the SCI_DEBUG output from $out_clean"
    echo " Now exiting...."
    exit
  fi

}

#______________________________________________________________________
#       Shift the cut utility field based on input tag
adjCutField()
{
  local tag=$1
  local -i field=$2

  if [[ "$tag" == "ComponentNodeStats" ]]; then
    field+=1
  fi

  echo "-f$field"
}
#______________________________________________________________________
#
printHeader()
{
  local -a desc="$1"
  shift
  local -a array=("$@")
  
  for i in "${array[@]}"; do
    if [[ "$i" == "Task*" ]]; then
      new="$desc$i"
      printf "\"%s\"", "$new"
    else
      printf "\"%s\"", "$i"
    fi
  done
  printf "\n"
}

#______________________________________________________________________
#
#______________________________________________________________________


main()
{

  if [[ -z $1 ]]; then
    echo "No parameter passed."
    exit
  else
    echo "Parameter passed = $1"
  fi

  #__________________________________
  # DEFAULTS:  Edit these
  declare -i startTimestep=3
  declare -i endTimestep=100           # timestep to extract elapsed time from.
  #__________________________________

  /bin/rm -rf .componentStats-*
  declare -x tmp=$(mktemp -d .componentStats-XXX)
  declare -x out_clean="$tmp/out.clean"
  declare -x out_clean2="$tmp/out.clean2"


  declare -a outputFiles
  mapfile -t outputFiles < <(grep --files-with-matches "TaskExec" "$@")

  if [[ ${#outputFiles[@]} -ne 0 ]]; then
    echo ""
    echo "__________________________________"
    echo " Now computing the averages for the SCI_DEBUG='ComponentNodeStats:+ or ComponentStats:+' variables, if they exist."
  fi

  #__________________________________
  #  Which SCI_DEBUG flag was set
  SCI_DEBUG=$(dialog --stdout --radiolist "Select what you want to extract" 10 35 40 "ComponentStats" "" off, "ComponentNodeStats" "" off )

  #__________________________________

  for out in "${outputFiles[@]}"; do
    if ( grep -q "Timestep $endTimestep" "$out" ); then
      echo "   working on $out"

      echo "$out" > "$tmp"/file
      awk -F: '/Parallel/&&/processes/ {printf "%i", $2}' "$out" > "$tmp"/nMPI


      # extract the output between the start and ending timesteps
      sed -n /"Timestep $startTimestep "/,/"Timestep $endTimestep "/p "$out" > "$tmp"/out.clean

      extract_tag_lines "$SCI_DEBUG"    #extract only exec/wait timelines


      me="$tmp"/out.clean2

      field=$( adjCutField "$SCI_DEBUG" 4 )
      grep TaskExec "$me"            | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/ave.taskExec
      grep TaskLocalComm "$me"       | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/ave.taskLocalComm
      grep TaskWaitComm "$me"        | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/ave.taskWaitComm
      grep TaskReduceCommTime "$me"  | cut -d: "$field" | xargs -r awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/ave.TaskReduceCommTime

      field=$( adjCutField "$SCI_DEBUG" 2 )
      grep TaskExec "$me"            | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/min.taskExec
      grep TaskLocalComm "$me"       | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/min.taskLocalComm
      grep TaskWaitComm "$me"        | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/min.taskWaitComm
      grep TaskReduceCommTime "$me"  | cut -d: "$field" | xargs -r awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/min.TaskReduceCommTime

      field=$( adjCutField "$SCI_DEBUG" 6 )
      grep TaskExec "$me"            | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/max.taskExec
      grep TaskLocalComm "$me"       | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/max.taskLocalComm
      grep TaskWaitComm "$me"        | cut -d: "$field" | awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/max.taskWaitComm
      grep TaskReduceCommTime "$me"  | cut -d: "$field" | xargs -r awk 'BEGIN {sum=0}; {sum=sum+$1}; END {print sum/NR}' > "$tmp"/max.TaskReduceCommTime

      paste -d "," "$tmp"/file "$tmp"/nMPI  "$tmp"/ave.taskExec "$tmp"/ave.taskLocalComm "$tmp"/ave.taskWaitComm "$tmp"/ave.TaskReduceCommTime >> "$tmp"/aveComponentTimes
      paste -d "," "$tmp"/file "$tmp"/nMPI  "$tmp"/min.taskExec "$tmp"/min.taskLocalComm "$tmp"/min.taskWaitComm "$tmp"/min.TaskReduceCommTime >> "$tmp"/minComponentTimes
      paste -d "," "$tmp"/file "$tmp"/nMPI  "$tmp"/max.taskExec "$tmp"/max.taskLocalComm "$tmp"/max.taskWaitComm "$tmp"/max.TaskReduceCommTime>> "$tmp"/maxComponentTimes

      paste -d "," "$tmp"/file "$tmp"/nMPI  "$tmp"/min.taskExec           "$tmp"/ave.taskExec           "$tmp"/max.taskExec \
                                            "$tmp"/min.taskLocalComm      "$tmp"/ave.taskLocalComm      "$tmp"/max.taskLocalComm \
                                            "$tmp"/min.taskWaitComm       "$tmp"/ave.taskWaitComm       "$tmp"/max.taskWaitComm \
                                            "$tmp"/min.TaskReduceCommTime "$tmp"/ave.TaskReduceCommTime "$tmp"/max.TaskReduceCommTime \
                                            >> "$tmp"/componentTimes
    else
      echo "---------------------$out did not run to completion"
    fi

  done  # loop over files

  #______________________________________________________________________
  #  header
  declare -a descHeader=("#file" "MPIprocs" "TaskExec" "TaskLocalComm" "TaskWaitComm" "TaskReduceCommTime")
  
  printHeader "avg" "${descHeader[@]}" >   aveComponentTimes
  sort -n -t"," "$tmp"/aveComponentTimes >>  aveComponentTimes
 # column -t "$tmp"/aveComponentTimes2 > aveComponentTimes
  more aveComponentTimes

  printHeader "min" "${descHeader[@]}" >   minComponentTimes
  sort -n -t"," "$tmp"/minComponentTimes >>  minComponentTimes
 # column -t "$tmp"/minComponentTimes2 > minComponentTimes
  more minComponentTimes

  printHeader "max" "${descHeader[@]}" >   maxComponentTimes
  sort -n -t"," "$tmp"/maxComponentTimes >>  maxComponentTimes
 # column -t "$tmp"/maxComponentTimes2 > maxComponentTimes
  more maxComponentTimes

  /bin/echo -e "#file,MPIprocs,minTaskExec,aveTaskExec,maxTaskExec"\
               ",minTaskLocalComm,aveTaskLocalComm,maxTaskLocalComm"\
               ",minTaskWaitComm,aveTaskWaitComm,maxTaskWaitComm"\
               ",minTaskReduceCommTime,aveTaskReduceCommTime,maxTaskReduceCommTime",> componentTimes
  sort -n -k2 "$tmp"/componentTimes >> componentTimes

  more componentTimes | column -s "," -t


#__________________________________
#     cleanup
#/bin/rm "$tmp"
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"

exit

#!/bin/bash

#______________________________________________________________________
#  This script, parses the SCI_DEBUG ExecTimes or WaitTimes output, between starting and ending timesteps
#  and for each task:
#
#ExecTimes:+
#--------------------------------------------------------------------------------
#                                                           Name:       Total:         Avg:         Min:         Max:    Max rank: Hist %% [  Q1 |  Q2 |  Q3 |  Q4 ]:  %%Load Imbalance
#                            ApplicationCommon::reduceSystemVars:      3.39 s       35.3 ms    0.0823 ms     0.114 s            11            66.7   0.0   0.0  33.3    69.0
#                            ApplicationCommon::updateSystemVars:     0.217 ms  2.26e+03 ns  1.63e+03 ns  3.17e+03 ns           24            27.1  41.7  20.8  10.4    28.6
#                                          Contact::initFriction:     0.138 s       1.44 ms    0.0248 ms      9.43 ms           94            81.2   9.4   4.2   5.2    84.7
#                             MPM::actuallyComputeStableTimestep:     0.304 ms  3.17e+03 ns  1.89e+03 ns  6.87e+03 ns           14            64.6  26.0   6.2   3.1    53.9
#                                        MPM::applyExternalLoads:      1.88 s       19.6 ms      1.93 ms      34.3 ms            9             3.1  45.8  34.4  16.7    42.7
#                           MPM::computeAndIntegrateAcceleration:      98.3 ms      1.02 ms     0.609 ms      3.78 ms           84            92.7   6.2   0.0   1.0    72.9
#                                MPM::computeCurrentParticleSize:      4.92 s       51.3 ms      5.93 ms      83.6 ms           48            12.5  14.6  49.0  24.0    38.6
#                                 MPM::computeGridVelocityForFTM:      3.25 s       33.9 ms    0.0419 ms      44.6 ms           81             2.1  10.4  25.0  62.5    24.1
#                                      MPM::computeInternalForce:      7.09 s       73.9 ms      50.5 ms     0.119 s            93            65.6   2.1   2.1  30.2    38.0
#                                  MPM::computeParticleGradients:      4.15 s       43.3 ms      29.2 ms      67.5 ms           21            65.6   1.0   2.1  31.2    35.9
#                                      MPM::computeSPlusSSPlusVp:      5.09 s         53 ms      31.7 ms      80.6 ms           18            35.4  32.3   8.3  24.0    34.3
#                                           <snip>              ^
#                                                           taskNameDelimiterCol
#______________________________________________________________________
#  AWk script that for each task computes the average over timesteps.
#  ave, min, and max values.  For eaxample:
#                     Name:       Total:         Avg:         Min:         Max:
# MPM::applyExternalLoads:      1.88 s       19.6 ms      1.93 ms      34.3 ms
#  awk fields                    $1 $2        $3  $4       $5 $6       $7   $8
#
#______________________________________________________________________
avgTaskTimes()
{
  awk -v task="$1" -v cutCol="$nTaskNameChar" '

  function convertToSeconds( x, units ){
    if( units == "ms"){
      return (x / 1.0e3)     # ms -> sec
    }

    if( units == "ns"){
      return (x / 1.0e9)     # ns -> sec
    }
    return x
  }
  #__________________________________

  BEGIN {                       # initialize the sums and the number of lines
    cutCol+=1;
    avg_sum=0;
    min_sum=0;
    max_sum=0;
    NL=0;
  }
  #__________________________________

  $0 ~ task {
      $0  = substr($0,cutCol)         # cut the first n chars of the line

      avg=convertToSeconds( $3, $4 )
      avg_sum = avg_sum + avg         # update the sum of the averages

      min=convertToSeconds( $5, $6 )
      min_sum = min_sum + min         # update the sum of the min

      max=convertToSeconds( $7, $8 )
      max_sum = max_sum + max         # update sum of the max
      NL  += 1                        # increment the number of lines
  };

  #__________________________________

  END {
    if ( NL > 0 ) {
      printf "%e, %e, %e, ", min_sum/NL, avg_sum/NL, max_sum/NL

      # printf "%s %e %e, %e, %e, \n", task, NL, min_sum/NL, avg_sum/NL, max_sum/NL  # debugging
    }
  }' "$out_clean3"
}
#______________________________________________________________________
#  This function looks at the first line of the clean output and finds
#  the column that has the delimiter ":" after the task name
findTaskNameDelimiterCol()
{
  outfile="$tmp"/taskNameDelimiterCol

  awk -v outfile="$outfile" '
  BEGIN{
    NR==1                                  # only examine the first line
  }
  {
    for (i = length($0); i >= 1; i--) {    # search output in reverse
      if( substr($0, i, 1) == ":"){
        printf( "%i\n", i ) > outfile
        exit
      }
    }
  }' "$out_clean3"
}

#______________________________________________________________________
#         Parse the output and clean out everything except
#         the lines assocated with ExecTimes/WaitTimes

extract_tag_lines()
{
  local tag=$1

  # extract all text between ExecTimes or WaitTimes
  sed -n '/'"$tag"'/,/^$/p'  "$out_clean"    >& "$out_clean2"

  # delete 2 line after
  sed '/'"$tag"'/,+2'D       "$out_clean2"  >& "$out_clean3"

   # delete any blank lines
  sed -i '/^$/d'              "$out_clean3"
}

#______________________________________________________________________
#         Parses the output and find all the task names
#    "            ICE::accumulateMomentumSourceSinks:     0.175 s   "
findTaskNames()
{
  # Find the name of each task
  cut -c 1-"$nTaskNameChar" "$out_clean3" >& "$tmp"/taskNames

  #  remove everthing after the last colon
  rev "$tmp"/taskNames | awk -F : '{for (i=2; i<NF; i++) printf $i ":"; print $NF}' | rev > "$tmp"/taskNames.1

  #  sort and remove dupilicate names
  cat "$tmp"/taskNames.1 | sort |uniq > "$tmp"/taskNames.clean

  # remove whitespace before taskname
  sed -i 's/^ *//g' "$tmp"/taskNames.clean


  # Load task names into an array
  mapfile -t taskNames < "$tmp/taskNames.clean"
}

#______________________________________________________________________
#       Allow the user to limit which tasks to analyze.  Default is all
#       This overwrites the orginal taskNames array
userInput_whichTasks()
{
  local -a taskNameList                   # create an array with list of tasks all enabled

  for task in "${taskNames[@]}"; do
    taskNameList+=( "$task" )             # be careful.  No additional spaces
    taskNameList+=( "on" )
  done

 mapfile -t taskNames < <(  dialog --stdout \
                                   --no-items \
                                   --separate-output \
                                   --checklist "Select the tasks to analyze." 40 80 80 "${taskNameList[@]}"  )
 echo "${taskNames[@]}"
}

#______________________________________________________________________
#            Print the task names and headers to the output file
printHeader()
{
  local out="$1"
  declare -i nTasks=${#taskNames[@]}

  for ((i = 0 ; i < nTasks ; i++)); do
    printf "\"%s-%s\", " "$i" "${taskNames[i]}" >> "$out"
  done

  printf "\n# nMPI, " >> "$out"

  for ((i = 0 ; i < nTasks ; i++)); do
    printf "%i-(min, avg, max), " "$i">> "$out"
  done

  printf "\n"  >> "$out"
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


  /bin/rm -rf .avgTaskTimes-*
  declare -x tmp=$(mktemp -d .avgTaskTimes-XXX)

  #__________________________________
  #     get from the user which times they want to parse
  declare -a exec_wait_tag
  mapfile -t exec_wait_tag < <(dialog --stdout \
                                      --separate-output \
                                      --checklist "Select what you want to extract" 10 35 40 "ExecTimes" "" off, "WaitTimes" "" off )
  #__________________________________
  #     Find the files that have the taskWait or taskExec times
  echo "exec_wait_tag: " "${exec_wait_tag[@]}"

  declare -a outputFiles
  mapfile -t outputFiles < <( printf -- '%s\n' "${exec_wait_tag[@]}"        | \
                              xargs -I {} grep --files-with-matches {} "$@" | \
                              sort -u )

  if [[ ${#outputFiles[@]} -ne 0 ]]; then
    echo ""
    echo "__________________________________"
    echo " Now computing the min, avg, max for all tasks reported by SCI_DEBUG='ExecTimes:+ or WaitTimes:+'"
  fi

  echo "${outputFiles[@]}"

  declare -x out_clean="$tmp/out.clean"
  declare -x out_clean2="$tmp/out.clean2"
  declare -x out_clean3="$tmp/out.clean3"

  declare -xi nTaskNameChar=-1            # column number where the task name ends
  declare -i startTimestep=3
  declare -i endTimestep=100

  declare -a outFilenames                 # array containing all the processed and parsed output filenames
  declare -ax taskNames                   # array containing all task names

  #__________________________________
  
  for tag in "${exec_wait_tag[@]}"; do

    firstPass=true

    for out in "${outputFiles[@]}"; do
      if ( grep -q "Timestep $endTimestep" "$out" ); then
        echo "   working on $tag, $out"

        outFilename="$tmp/task$tag-$out"      # form array of processed file names

        outFilenames+=( "$outFilename" )

        touch "$outFilename"

                                              # find number of MPI processes
        nMPI+=( $( awk -F: '/Parallel/&&/processes/ {printf "%i", $2}' "$out" ) )

                                              # extract all text between starting and ending timesteps
        sed -n /"Timestep $startTimestep "/,/"Timestep $endTimestep "/p "$out" > "$tmp"/out.clean

        extract_tag_lines "$tag"              # extract only exec/wait timelines

        findTaskNameDelimiterCol              # find the column that has the delimiter :

        nTaskNameChar=$(more "$tmp"/taskNameDelimiterCol)

        findTaskNames                         # find the task names

        if $firstPass; then
          userInput_whichTasks

          printHeader "$outFilename"
        fi

                                        # loop over all timesteps and tasks and compute sum and averages
        printf "%i, " "${nMPI[-1]}" >> "$outFilename"
        for task in "${taskNames[@]}"; do
          avgTaskTimes "$task">> "$outFilename"
        done

      else
        echo  "---------------------$out did not run to completion"
      fi

      firstPass=false
      i=0
    done
  done

  #__________________________________
  # loop over all the output files and combine them into one file
  for tag in "${exec_wait_tag[@]}"; do
    out="task_$tag"

    /bin/rm -f "$out" "$out"_sorted > /dev/null

    #  only examine either the ExecTimes or WaitTime files
    declare -a subset
    mapfile -t subset< <( printf '%s\n' "${outFilenames[@]}" | grep "$tag" )

    head -n 2 "${subset[0]}" >> "$out"

    for tf in "${subset[@]}"; do
      tail -n 1 "$tf" >> "$out"
      printf "\n" >> "$out"
    done

    sort -n -k1 "$out" > "$out"_sorted
  done
}

#5.329794e-05, 2.597594e-03, 5.142990e-03,
#1234567890123456789012345678901234567890

#__________________________________
#     cleanup
#/bin/rm "$tmp"

#______________________________________________________________________
#______________________________________________________________________

main "$@"

exit

#!/bin/bash
#______________________________________________________________________
#  This script parses the output file from sus and
#  generates a data file used for scaling plots
#  usage:
#      extractScalingData  <sus_output.1, sus_output_2, sus_output_3 >
#
#______________________________________________________________________


main()
{
  declare -a outputFiles=( "$@" )

  if [[ ${#@} -eq 0 ]]; then
    echo "    Usage:  extractScalingData  <sus_output.1, sus_output_2, sus_output_3 >"
    exit
  fi

  #__________________________________
  # DEFAULTS:  Edit these
  declare -i startTimestep="3"
  declare -i endTimestep="100"           # timestep to extract elapsed time from.
  #__________________________________

  # make work directory
  /bin/rm -rf .scaling-*
  declare -x tmp=$(mktemp -d .scaling-XXX)
  touch "$tmp"/data

  for out in "${outputFiles[@]}"; do

    if ( grep -q "Timestep $endTimestep" "$out" ); then
      echo "   working on $out"

      echo "$out" > "$tmp"/file
      awk -F: '/Parallel/&&/processes/ {printf "%i", $2}' "$out" > "$tmp"/nMPI

      grep --max-count 1 "Timestep $startTimestep" "$out" | awk -F "=" '{print $4}' | tr -d "[:alpha:]" > "$tmp"/startTime
      grep --max-count 1 "Timestep $endTimestep"   "$out" | awk -F "=" '{print $4}' | tr -d "[:alpha:]" > "$tmp"/endTime

      paste -d " " "$tmp"/file "$tmp"/nMPI "$tmp"/startTime "$tmp"/endTime >> "$tmp"/data

    else
      echo "---------------------$X did not run to completion"
    fi

  done
  #__________________________________
  # compute the average mean time per timestep

  declare -i n=endTimestep-startTimestep
  mesg="#Computing average mean time per timestep for $n timesteps."
  echo "$mesg"
  
  printf '%s\n' "$mesg" > "$tmp"/data2
  echo "#file MPIprocs startTime endTime aveMean" >>"$tmp"/data2

  awk -vnSteps=$n '{print ($4-$3)/nSteps}' "$tmp"/data >> "$tmp"/aveMeanTime

  paste -d " " "$tmp"/data "$tmp"/aveMeanTime >> "$tmp"/data2
  sort -n -k2 "$tmp"/data2| column -t >scalingData

  more scalingData


  /bin/rm -rf "$tmp"
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"

exit

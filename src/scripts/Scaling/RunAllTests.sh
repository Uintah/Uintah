#!/bin/bash 

SIZES="128 256 512"

for s in $SIZES; do

  OUTPUT_BASE="outputs_arm"


  if [ -n "$OUTPUT_BASE" ] && [ ! -e "$OUTPUT_BASE" ]; then
    mkdir -p "$OUTPUT_BASE" 
  fi


  OUT="out.C1-R$s"

  ./scalingRuns.sh --config 1 --res "$s" --nodes "arm" >& "$OUTPUT_BASE"/"$OUT"
done

exit

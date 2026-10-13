#!/bin/bash
#______________________________________________________________________
#   Generate movies from DataAnalysis:lineExtract
#
#   usage:
#   src/scripts/lineExtractMovies --uda <UDA> --config <PLOT CONFIG> --line name
#
#______________________________________________________________________

if ! options=$(getopt --alternative \
                      --name lineExtractMovies \
                      --options "" \
                      --longoptions uda:,config:,line: \
                      -- "$@"); then
  echo "Usage: lineExtractMovies --uda DIR --config FILE --line NAME  " >&2
  exit 1
fi

eval set -- "$options"

line="NULL"

while true; do
  case "$1" in
    --uda)      uda="$2";    shift 2 ;;
    --config)   config="$2"; shift 2 ;;
    --line)     line="$2";   shift 2 ;;
    --)         shift; break ;;
    *)         echo "Internal error: unhandled option $1" >&2; exit 1 ;;
  esac
done


# bulletproofing
if [[ -z $uda || -z $config ]]; then
  echo "--uda and --config are required" >&2
  exit 1
fi


# src = parent of the directory this script lives in (resolves symlinks)
scriptDir=$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )
src=$( dirname "$scriptDir" )


#______________________________________________________________________
#   Process the lineExtract files into a time series
"$src"/scripts/combineLineExtactData.sh -d x "$uda"/"$line"
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "ERROR(lineExtractMovies): combineLineExtactData.sh failed (return code $rc)" >&2
  exit $rc
fi


#   Generate line plots from the timeseriesl using the config file
"$src"/scripts/plotLineExtractProfiles.py -n 10 \
                                        --uda "$uda" \
                                        --config "$config" \
                                        --line-name "$line"
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "ERROR(lineExtractMovies): plotLineExtractProfiles.py failed (return code $rc)" >&2
  exit $rc
fi

cd "$uda"/"$line"/L-0/plots
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "ERROR(lineExtractMovies): cannot cd to $uda/$line/L-0/plots (return code $rc)" >&2
  exit $rc
fi

#__________________________________
# loop over all pages and create movie(s)

mapfile -t these < <(find * -name "*.png" | cut -c 1-2 | sort -u)

for val in "${these[@]}"; do
  echo "$val"

  "$src"/scripts/ppmToMpeg.sh "$val" < /dev/null
done

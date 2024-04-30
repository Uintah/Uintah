#!/bin/bash

source "$( realpath "$0" | xargs dirname )/plotUtils"
#______________________________________________________________________
#  This script parses sus output and plots it using gnuplot.
#  This requires gnuplot 3.8 or greater.
#______________________________________________________________________


usage()
{
  echo "Usage: $0"
  echo ""
  echo "OPTIONS:"
  echo "     --file          <output file> [ manditory]  Name of the outputfile to parse"
  echo "     --timestepRange <low:high>    [ optional ]  x-axis range"
  echo "     --pdf                         [ optional ]  output a pdf file"
  exit 1
}

#______________________________________________________________________

main()
{
  enableTrapping

  /bin/rm -rf scraps >&/dev/null
  mkdir scraps >&/dev/null

  commandExists gnuplot

  #__________________________________
  # parse arguments
  options=$( getopt --name "plotStats" --options="h"  --longoptions=help,file:,timestepRange:,pdf -- "$@" )

  if [ $? -ne 0 ] || [ $# -eq 0 ] ; then
    echo "Incorrect option provided"
    usage
    exit 1
  fi

  # set is to preserve white spaces and punctuation
  eval set -- "$options"

  #__________________________________
  # parse inputs
  hardcopy="N"
  tsRange=":"
  out="null"

  while true; do
    case "$1" in
      -h|--help)
        usage
        ;;
      --file)
        shift
        out="$1"
        ;;
     --timestepRange)
        shift
        tsRange="$1"
        ;;
     --pdf)
        hardcopy="Y"
        ;;
     --)
        shift
        break
        ;;
     esac
     shift
  done

  if [ -n "$out" ] && [ ! -e "$out" ]; then
    end_die "  The uda output file (${out}) was not found, now exiting"
  fi


  #__________________________________
  # Make a copy of the output file and remove
  # the extra spaces
  /bin/rm -rf scraps >& /dev/null
  mkdir scraps >&/dev/null

  cat "$out" | tr -s ' ' >& scraps/out.clean
  cd scraps || exit

  #   only timesteps with Memory
  grep Time= out.clean |grep Mem  >& out_timesteps

  #______________________________________________________________________
  # Parse the output file.  This is gross
  #   1       2      3    4       5    6      7       8     9    10     11    12
  #Timestep 188290 Time=884.702 Next delT=0.00462586 Wall Time=86393.5 EMA=0.454072 ETC=03:08:37 Memory Used=188.10 MBs (avg) 256.43 MBs (max on rank: 17 )

  echo "  Now extracting the data from $out"

  awk -F'=| ' '{ print $2 } ' out_timesteps >& timestep
  awk -F'=| ' '{ print $4 }'  out_timesteps >& physicalTime
  awk -F'=| ' '{ print $7 }'  out_timesteps >& delT
  awk -F'=| ' '{ print $10 }' out_timesteps >& elapsedTime
  awk -F'=| ' '{ print $12 }' out_timesteps >& meanTime

  awk -F'Memory Used' 'split($2,a,"=| "){ print a[2] }' out_timesteps  > memAve
  awk -F'Memory Used' 'split($2,a," ")  { print a[4] }' out_timesteps  > memMax

  # compute the time per timestep
  awk 'NR>1{ printf("%.15f\n",$1-p)} {p=$1}' elapsedTime >& timePerTimestep
  echo "?" >> timePerTimestep

  # paste the columns together
  paste -d "  " timestep elapsedTime meanTime timePerTimestep physicalTime delT memAve memMax>& data

  # select timestepss requested
  pruneTimesteps "$tsRange" "data"

  # extract the meta data from the header of the output
  declare -A metaData
  parseMetaData out.clean metaData

  echo "  Parsed meta data:"
  echo "  ${metaData[date]}, ${metaData[machine]}, ${metaData[procs]} MPI ranks, ${metaData[threads]} Threads, uda: ${metaData[uda]}"
  echo "  Done extracting data"

#______________________________________________________________________
#  create the gnuplot script

  touch  gp

  # set the gnuplot terminal
  setTerminal "$hardcopy" "gp" "plotStat.ps"

cat >> gp << fin

#__________________________________
# compute stats this version of gnuplot is > 3.8

if ( strstrt(GPVAL_COMPILE_OPTIONS,"+STATS") ) {
  print " Computing the statistics";
} else {
  print " This version of gnuplot does not support computing statistics.";
  print " Now exiting...."
  exit
}

E_records=0                                 # default for max memory

#  Data file columns
#     1        2          3            4            5          6    7      8
# timestep elapsedTime meanTime timePerTimestep physicalTime delT memAve memMax
stats 'data' using 5 name "A" nooutput;
stats 'data' using 3 name "B" nooutput;
stats 'data' using 4 name "C" nooutput;
stats 'data' using 7 name "D" nooutput;
stats 'data' using 8 name "E" nooutput;
#show variables all                          # debugging

set multiplot
set size 1.0,0.33 # for three plots
set autoscale
set grid xtics ytics


#__________________________________ TOP
#     Time related
set size   1.0, 0.33
set origin 0.0, 0.66

set title "${metaData[date]}, ${metaData[machine]}, ${metaData[procs]} MPI ranks, ${metaData[threads]} Threads, uda: ${metaData[uda]}"

set xlabel "Elaspsed Time [s]" offset 0,+0.5
set ylabel 'Delt'           textcolor lt 1 offset +2,0
set y2label 'Physical Time' textcolor lt 2 offset -2,0
set y2tics
set autoscale xfix

set y2range[A_mean - 2*A_stddev:A_mean + 2*A_stddev]

plot 'data' using 2:6           t 'Delt' with lines,\
     'data' using 2:5 axes x1y2 t 'Physical Time' with lines

#__________________________________ Middle
#     Mean time per timestep
set origin 0.0,0.33
set title ''
set xlabel "Timestep" offset 0,+0.5
set ylabel 'Mean Time/timestep'           textcolor lt 1  
set y2label 'Time per timestep [sec]'     textcolor lt 2  
set y2tics
set autoscale xfix

ymax=(B_up_quartile + B_stddev)
if ( (C_up_quartile + C_median) > ymax ) {
  ymax = (C_up_quartile + C_median)
}

set yrange[ 0:ymax]
set y2range[0:ymax]

plot 'data' using 1:3           t 'meanTime/timestep' with lines, \
     'data' using 1:4 axes x1y2 t 'time/timestep'     with lines


#__________________________________Bottom
#     Memory plots
set origin 0.0,0.0
set title ''
set xlabel "Elapsed Time [s]" offset 0,+0.5
set ylabel  "Ave memory usage [MB]"     textcolor lt 1
set yrange[ D_min - D_stddev:D_max + D_stddev]

set autoscale xfix

if( E_records ){                              # if the outputfile contains max memory stats
  set yrange[ D_min - D_stddev:E_max + E_stddev]
  set y2label 'Max memory usage [MB]'     textcolor lt 2
  set y2range[D_min - D_stddev:E_max + E_stddev]

  plot 'data' using 2:7   t 'ave' with lines, \
       'data' using 2:8   t 'max' with lines
} 
else{                                       #  if the outputfile doesn't contain max stats
  unset y2label
  set link y2
    
  plot 'data' using 2:7   t 'ave' with lines
}

set nomultiplot

if ( "$hardcopy" eq "N") {
  pause -1 "  Hit return to exit"
}
fin
#______________________________________________________________________


  #__________________________________
  # plot it up

  gnuplot gp

  create_pdf "$hardcopy" "plotStat"

  #__________________________________
  # clean up
  /bin/rm -rf scraps

  exit 0
}

#______________________________________________________________________
#______________________________________________________________________

main "$@"

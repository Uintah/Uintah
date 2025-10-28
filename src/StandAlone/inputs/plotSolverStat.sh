#!/bin/bash

source "$( realpath "$0" | xargs dirname )/plotUtils"
#______________________________________________________________________
#  This script parses sus output and plots relevant hypre solver quantities
#
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
}
#______________________________________________________________________

parseOutputFile()
{
  awk --lint -F'Timestep|completed in|max_rhs before solve' '

    #__________________________________
    BEGIN {
      p=0
      timestep    =0;
      outerIter   =0;
      max_rhs     =0;
      min_rhs     =0;
      solverTime  =0;
      solverIters =0;
      printHeader =1;
      totalSolverTime     =0;
      solverMovingAvgTime =0;
    };

    #__________________________________
    $0 ~ "Timestep"{
      split( $2, a,/ / );
      timestep =a[2]
      next
    };

    #__________________________________
    $0 ~ "completed in"{

      split( $2, a, " " );
      #{for(i in a) print "  a[",i,"] ", a[i]}          # for debugging

      totalSolverTime     = a[1]
      solverTime          = a[5]
      solverMovingAvgTime = a[8]
      solverIters         = a[10]
      next
    };

    #__________________________________
    $0 ~ "max_rhs before solve"{

       split( $1, a, " " );
       split( $2, b, " " );

       #{for(i in a) print "  a[",i,"] ", a[i]}          # for debugging
       #{for(i in b) print "  b[",i,"] ", b[i]}          # for debugging

       outerIter = a[3]
       max_rhs   = b[1]
       min_rhs   = b[4]
       p=1
    };
    #__________________________________
    p{
      #printf "#%i timestep %i iter %i max_rhs %e min_rhs %e", NR,timestep, outerIter, max_rhs, min_rhs
      #printf " totalSolverTime %e solverTime %e movingAvg %e\n", totalSolverTime, solverTime, solverMovingAvgTime

      if( printHeader == 1 ){
        printf "#timestep, outerIter, solverIters, solverMovingAvgTime, solverTime, totalSolverTime, max_rhs, min_rhs\n" > "data";
        printHeader = 0;
      }

      out = "data"
      if( outerIter > 1 ){        # if the number of outer iterations is > 1 then write to data2
        out = "data2"
      }

      printf "%i %i %i %e %e %e %e %e\n", timestep, outerIter, solverIters, solverMovingAvgTime, solverTime, totalSolverTime, max_rhs, min_rhs > out;
      p=0
    }
    END {
      close ("data")

      if ( system("test -f data2") == 0) {
        close ("data2")
      }
    }' "$1"
}
#______________________________________________________________________
main()
{

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
  declare -i plotData2=0

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

  awk '/Timestep|Solve of | Outer iteration/ { print }' "$out" > scraps/out.clean

  sed -n '/Timestep 0/q;p' "$out" > scraps/out.header

  cd scraps || exit

  #______________________________________________________________________
  # Parse the output file.  This is gross
  echo "  Now extracting the data from $out"

  parseOutputFile "out.clean"

  # select timestepss requested
  pruneTimesteps "$tsRange" "data"
  pruneTimesteps "$tsRange" "data2"
  if [[ -s "data2" ]]; then
    plotData2=1
  fi


  # extract the meta data from the header of the output
  declare -A metaData
  parseMetaData "out.header" metaData

  echo "  Parsed meta data:"
  echo "  ${metaData[date]}, ${metaData[machine]}, ${metaData[procs]} MPI ranks, ${metaData[threads]} Threads, uda: ${metaData[uda]}"
  echo "  Done extracting data"

#______________________________________________________________________
#  create the gnuplot script
#______________________________________________________________________

  touch  gp

  # set the gnuplot terminal
  setTerminal "$hardcopy" "gp" "plotSolverStat.ps"


cat >> gp << fin

#__________________________________
# compute stats this version of gnuplot is > 3.8
if ( strstrt(GPVAL_COMPILE_OPTIONS,"+STATS") ) {
  print "Computing the statistics of the mean time per timestep";
} else {
  print " This version of gnuplot does not support computing statistics.";
  print " Now exiting...."
  exit
}

# compute statistics of mean time and scale the results
#    1         2         3                  4             5                6            7        8
##timestep, outerIter, solverIters, solverMovingAvgTime, solverTime, totalSolverTime, max_rhs, min_rhs
stats 'data' using 3 name "A";
stats 'data' using 4 name "B";
stats 'data' using 5 name "C";
stats 'data' using 6 name "D";

set multiplot
set size   0.5, 0.5

set grid xtics ytics

unset title
set xlabel 'Timesteps'
set ylabel 'Time [s]'        textcolor lt 2
set yrange[B_mean - 2*B_stddev:B_mean + 2*B_stddev]

set origin 0.0, 0.0
plot 'data' using 1:4   t 'solver moving average /timestep' with lines

#__________________________________
set title "${metaData[date]}, ${metaData[machine]}, ${metaData[procs]} MPI ranks, ${metaData[threads]} Threads, uda: ${metaData[uda]}" offset screen 0.25,0.03 noenhanced
set ylabel 'Solver Time [s]' textcolor lt 2

unset xlabel
set yrange[C_mean - 2*C_stddev:D_mean + 2*D_stddev]

set origin 0.0, 0.5

plot 'data' using 1:5  t 'solve Time ' with lines, \
     'data' using 1:6  t 'total solver Time ' with lines

#__________________________________
unset title
set ylabel 'Solver iterations'
set y2label'outer iterations'
set ytics  1       # integer labels
set y2tics 1
set yrange[A_mean - 2*A_stddev:A_max + 1]
set origin 0.5, 0.0

if( "$plotData2" == 0){
  plot 'data' using 1:3  t 'solver iterations' with lines

}
else{
  plot  'data'  using 1:3           t 'solver iterations' with lines,\
        'data2' using 1:2 axes x1y2 t 'outer iterations'

}

#__________________________________
set ylabel 'max rhs'
unset y2label
set ytics auto
set autoscale
set origin 0.5, 0.5
set size   0.5, 0.25

plot 'data' using 1:7 t 'before solve' with lines

#__________________________________

set ylabel'min rhs after solve'
set origin 0.5, 0.75

plot 'data' using 1:8 t 'after solve' with lines
set nomultiplot



if ( "$hardcopy" eq "N") {
  pause -1 "  Hit return to exit"
}
fin
#______________________________________________________________________


  #__________________________________
  # plot it up

  gnuplot gp

  create_pdf "$hardcopy" "plotSolverStat"

  #__________________________________
  # clean up
  /bin/rm -rf scraps

  exit 0
}

#______________________________________________________________________
#______________________________________________________________________
main "$@"

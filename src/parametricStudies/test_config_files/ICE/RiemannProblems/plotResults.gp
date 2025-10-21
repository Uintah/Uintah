susResults = ARG1
exactSol   = ARG2
resolution = ARG3
myTitle    = ARG4
hardcopy = 1

#__________________________________
#   define the terminal
if( hardcopy ){
  fileName = sprintf( "comparisonPlots_%s.eps",resolution )
  set terminal postscript color solid "Times-Roman" 9 portrait
  set output fileName
} else {
  set terminal x11 enhanced font "arial,12"  size 1800,1400
}

set autoscale
set multiplot  layout 4,1

set xrange [0:1]
set pointsize 0.4
set grid

#__________________________________
#   pressure
set title myTitle
set ylabel "Pressure"
plot  susResults     using 1:4 t 'ICE' with linespoints ,\
      exactSol       using 1:4 t 'exact' with lines

#__________________________________
#   Temperature

unset title
set ylabel "Temperature"
plot  susResults     using 1:5 t 'ICE' with linespoints,\
      exactSol       using 1:5 t 'exact' with lines
#__________________________________
#  velocity x-component

set ylabel "Velocity"
plot  susResults     using 1:3 t 'ICE'   with linespoints,\
      exactSol       using 1:3 t 'exact' with lines
#__________________________________
#   density

set ylabel "Density"
set xlabel "X"
plot  susResults     using 1:2 t 'ICE'  with linespoints,\
      exactSol       using 1:2 t 'exact' with lines

set nomultiplot

pause -1

if( hardcopy) {
  c1 = sprintf( 'ps2pdf -dEPSCrop %s; rm %s', fileName, fileName )
  system( c1 )
}


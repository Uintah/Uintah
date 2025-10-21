data=ARG1
hardcopy = 1

#__________________________________
#   define the terminal 
if( hardcopy ){     
  cmd = sprintf( "basename %s ", data )                           # eliminate the path
  tmp = system( cmd )

  cmd = sprintf( "echo %s | rev | cut -f 2- -d '.' | rev", tmp )   # remove only the last extension Gross
  fileName = system (cmd)

  set terminal postscript color solid "Times-Roman" 9 portrait
  set output fileName
} else {
  set terminal x11 enhanced font "arial,12"  size 1800,1400
}

set autoscale
set multiplot  layout 4,1

set y2tics
set xrange [0:1]

set pointsize 0.3
set grid

#__________________________________
#   pressure

set ylabel "Pressure"
plot  data     using 1:4 t 'exact' with lines

#__________________________________
#   Temperature

set ylabel "Temperature"
plot  data     using 1:5 t 'exact' with lines

#__________________________________
#  velocity x-component

set ylabel "Velocity"
plot  data     using 1:3 t 'exact'  with lines

#__________________________________
#   density

set ylabel "Density"
plot  data    using 1:2 t 'exact'  with lines 

set nomultiplot

pause -1

if( hardcopy) {
  c1 = sprintf( 'ps2pdf -dEPSCrop %s; rm %s', fileName, fileName )
  system( c1 )
}


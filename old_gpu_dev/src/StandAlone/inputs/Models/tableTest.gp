#__________________________________
#  cleanup the mess
#__________________________________
!/bin/rm -f gp.dat, tmp.dat1

#__________________________________
#  gnuplot script that plots table values
#__________________________________
set ytics
set xtics
set y2tics
set mxtics
set mytics
set grid xtics ytics
set pointsize 0.5
set title "Table sanity test"
set xlabel "Mixture fraction"
set autoscale
set xrange [0:1.0]

#  rip out "[" "]" data
!/bin/rm gp.dat
!grep "\[" data >& tmp.dat1
!sed 's/\[//g' tmp.dat1 | sed 's/\]//g' >gp.dat
#__________________________________
#  gamma
#__________________________________
set terminal x11 1
#set terminal postscript color "Times-Roman" 9
#set output "thermoProps.ps"
set multiplot
set size 1.0,0.51  
set origin 0.0,0.0
set ylabel "gamma"
plot  'gp.dat' using 1:5 t 'table'

#__________________________________
#   cv
#__________________________________
set origin 0.0,0.5
set ylabel " Specific Heat J/kg-K"
plot   'gp.dat' using 1:6 t 'table'

set nomultiplot 
#__________________________________
#  table temperature
#__________________________________
set terminal x11 2
#set terminal postscript color "Times-Roman" 9
#set output "temperature.ps"
set size 1.0,1.0
set origin 0.0,0.0
set ylabel "Temperature K"
plot  'gp.dat' using 1:4      t 'Table'
#__________________________________
#   table pressure
#__________________________________
set terminal x11 3
#set terminal postscript color "Times-Roman" 9
#set output "pressure.ps"
set size 1.0,1.0
set origin 0.0,0.0
set ylabel "Pressure [Pa]"
plot  'gp.dat' using 1:8      t 'Table Values: p = ((gamma -1 ) cv density temperature)'

#__________________________________
#   table density
#__________________________________
set terminal x11 4
#set terminal postscript color "Times-Roman" 9
#set output "rho.ps"
set size 1.0,1.0
set origin 0.0,0.0
set ylabel "Density [kg/m^3]"
plot  'gp.dat' using 1:7      t 'table'


#__________________________________
#   (gamma -1 ) * cv
#__________________________________
set terminal x11 5
#set terminal postscript color "Times-Roman" 9
#set output "thermo.ps"
set size 1.0,1.0
set origin 0.0,0.0
set ylabel "(gamma -1) cv"
plot  'gp.dat' using 1:9      t 'table'


#__________________________________
#  (density * temperature0
#__________________________________
set terminal x11 6
#set terminal postscript color "Times-Roman" 9
#set output "physical.ps"
set size 1.0,1.0
set origin 0.0,0.0
set ylabel "density * temperature"
plot  'gp.dat' using 1:10      t 'table'

#__________________________________
#  co2 & h2o
#__________________________________
#set terminal x11 7
set terminal postscript color "Times-Roman" 9
set output "concentrations.ps"
set size 1.0,1.0
set origin 0.0,0.0
set ylabel "Concentrations"
plot  'gp.dat' using 1:11      t 'co2',\
      'gp.dat' using 1:12      t 'h2o'
pause -1 "Hit return to continue"


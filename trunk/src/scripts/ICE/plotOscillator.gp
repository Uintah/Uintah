
set grid xtics ytics
set pointsize 1

set autoscale
#set terminal x11 1
set terminal  postscript "Times-Roman" 12
set output "waterOscillator.ps"

data = 'WaterAirOscillator.uda/P/L-0/i20_j10_k0'


set multiplot
set size 0.51,0.51
set origin 0.0,0.5  

set title "Probe point in the center of the water piston"

#__________________________________
set ylabel "Pressure [Pa]"
set xlabel "Time [s]"
set xrange[1e-2:1.0]

plot   data      using 4:5            t ''  w lines,

#__________________________________
set title ""
set origin 0.5,0.0
set ylabel "Temperature[K]"
set xlabel "Time [s]"

plot   data      using 4:7            t ''  w lines,

#__________________________________
set origin 0.5,0.5
set ylabel "Vel_CC.x [m/s]"
set xlabel "Time [s]"

plot   data      using 4:9            t ''  w lines,

#__________________________________
set origin 0.0,0.0
set ylabel "delP [Pa]"
set xlabel "Time [s]"

plot   data      using 4:8            t ''  w lines,

set nomultiplot   
pause -1 "Hit return to continue"


!ps2pdf waterOscillator.ps
!/bin/rm waterOscillator.ps
!pdfcrop waterOscillator.pdf

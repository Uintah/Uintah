set terminal x11 1
set terminal postscript color solid "Times-Roman" 10
set output "StatisticVerificationPlots.ps"

set title "Scalar-f =  A * sin(omega * time),  A = 10, omega = 100"

set multiplot
set size 1.0,0.24

set origin 0.0, 0.75
plot 'out-f' using 1:2 with linespoints t 'scalar-f'

unset title
set origin 0.0, 0.5
plot 'out-variance' using 1:2 with linespoints t 'variance-f'


set origin 0.0, 0.25
plot 'out-skewness' using 1:2 with linespoints t 'skewness-f'


set origin 0.0, 0.0
set xlabel "Physical Time [s]"
set label "Data sampled from a single cell in the center of the domain" at screen 0.1,0.1
plot 'out-kurtosis' using 1:2 with linespoints t 'kurtosis-f'

set nomultiplot

pause -1 "hit return to exit'

!ps2pdf StatisticVerificationPlots.ps
!/bin/rm StatisticVerificationPlots.ps

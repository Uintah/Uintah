# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "orderAccuracy.ps"

set autoscale
set grid xtics ytics

#title
#xlabel
#ylabel
#label

plot 'L2norm.dat' using 1:2 t 'LODI outflow |R|' with linespoints pt 7

!ps2pdf orderAccuracy.ps

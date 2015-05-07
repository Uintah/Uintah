
# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "divQ_Error.ps"
 
set autoscale
set grid xtics ytics
set logscale x
set logscale y

set label "Test case: partially developed Methane flame" at screen 0.2, 0.2
set label "1000 Rays/cell" at screen 0.2, 0.15

#title
#xlabel
#ylabel


!paste L2norm.dat Lnorms > Lnorms.all

set pointsize 1
plot 'Lnorms.all' using 1:4 t ''

!ps2pdf divQ_Error.ps


# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "BoundFlux_Error.ps"
 
set autoscale
set grid xtics ytics
set logscale x
set logscale y

set label "Burns & Christon Benchmark" at screen 0.2, 0.2
set label "Comparison against CPU code run with 1000 Boundary Flux Rays" at screen 0.2, 0.15

#title
#xlabel
#ylabel

!paste L2norm.dat L1 > L1.all
!paste L2norm.dat L2 > L2.all
!paste L2norm.dat Linf >Linf.all

set pointsize 1
plot 'L2.all' using 1:2 t 'W Face',\
     'L2.all' using 1:3 t 'E ',\
     'L2.all' using 1:4 t 'S ',\
     'L2.all' using 1:5 t 'N ',\
     'L2.all' using 1:6 t 'B ',\
     'L2.all' using 1:7 t 'T ' with points pt 13 linecolor 1

!ps2pdf BoundFlux_Error.ps

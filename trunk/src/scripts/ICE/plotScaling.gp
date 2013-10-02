
set term x11

# uncomment below for post script output
#set terminal postscript color solid "Times-Roman" 18
#set output "scaling.ps"
set style line 1  lt 1 lw 0.6 lc 1
 
set autoscale
set logscale x
set logscale y
set grid xtics ytics

set title "XX Simulation"
set xlabel "Cores"
set ylabel "Average time per timestep [s]"

set xrange [8:1024]
set label "Averaged over XX timesteps" at screen 0.2,0.2
set label "Resolution: XX, XX patches" at screen 0.2, 0.15

set pointsize 1.0

plot 'sortedData'   using 2:5 with linespoints lw 2 pointsize 2 t''
     
!ps2pdf scaling.ps
!/bin/rm scaling.ps
!pdfcrop scaling.pdf

pause -1

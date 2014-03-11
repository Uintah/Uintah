
set term x11

# uncomment below for post script output
#set terminal postscript color solid "Times-Roman" 18
#set output "scaling.ps"
set style line 1  lt 1 lw 0.6 lc 1
 
set autoscale
set logscale x
set logscale y
set grid xtics ytics
#set xtics ("12" 12, "24" 24, "48" 48, "96 " 96, "192" 192, "384" 384, "768" 768, "1536" 1536)

set title "XX Simulation"
set xlabel "Cores"
set ylabel "Average time per timestep [s]"

set xrange [8:1024]
set label "Averaged over XX timesteps" at screen 0.2,0.2
set label "Resolution: XX, XX patches" at screen 0.2, 0.15

# generate the ideal scaling line
f1(x) = a1*x**b1               # define the function to be fit
a1 = 0.1; b1 = -1 
FIT_LIMIT=1e-6
fit f1(x) 'sortedData' using 2:5 via a1

set pointsize 1.0

plot 'sortedData'   using 2:5 with linespoints lw 2 pointsize 2 t'',\
     f1(x) t ''
     
!ps2pdf scaling.ps
!/bin/rm scaling.ps
!pdfcrop scaling.pdf

pause -1

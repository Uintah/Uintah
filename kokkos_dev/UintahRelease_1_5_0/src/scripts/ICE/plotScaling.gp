set term x11

# uncomment below for post script output
#set terminal postscript color solid "Times-Roman" 14
#set output "scaling.ps"
#set style line 1  lt 1 lw 0.6 lc 1
 
set autoscale
set logscale x
set logscale y
set grid xtics ytics

# kraken
set xtics ("12" 12, "24" 24, "48" 48, "96 " 96, "192" 192, "384" 384, "768" 768, "1536" 1536, "3072" 3072)


set title "Kraken: AR5D16 (260x160x160) \n Data Analysis OFF"
set xlabel "Cores"
set ylabel "Time per timestep (s)"

# generate the x_curvefit
f1(x) = a1*x**b1               # define the function to be fit
a1 = 0.1; b1 = -1 
FIT_LIMIT=1e-6


# mean time per timestep
#fit f1(x) 'sortedData' using 2:7 via a1
#plot 'scalingData' using 2:5 with linespoints t' ',\
#     f1(x) t ''
     
     
#mean time per timestep and pressure solve
fit f1(x) 'sortedData' using 2:7 via a1
plot 'sortedData' using 2:7 with linespoints t'Total ',\
     'sortedData' using 2:8 with linespoints t'Hypre Linear Solve ',\
     f1(x) t 'Ideal'
     
     
pause -1

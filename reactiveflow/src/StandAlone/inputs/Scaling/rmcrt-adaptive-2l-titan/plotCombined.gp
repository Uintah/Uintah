# uncomment below for post script output
#set terminal X11
set terminal postscript color solid "Times-Roman" 14
set output "scaling-2L-adaptive-rmcrt_DO.ps"

set style line 1  pt 2 lw 1.5 lc 2
set pointsize 2
 
set autoscale
set logscale x
set logscale y
set grid xtics ytics

set xtics ("16" 16, "32" 32, "64" 64, "128" 128, "256" 256, "512" 512, "1024" 1024, "2048" 2048, "4096" 4096, "8192" 8192, "16K" 16384, "32K" 32768, "65K" 65536, "131K" 131071, "262K" 262144)

set title  "RMCRT:CPU:DOUBLE\n Burns \& Christon Benchmark\n2-Level Adaptive RMCRT,    ROI: patch based\nOLCF-Titan System"
set xlabel "CPU Cores"
set ylabel "Mean Time Per Timestep (s)"
set label  "Unified MPI/threaded scheduler\n1 MPI proc & 16 threads per node\n100 rays per cell\nAveraged over 7 timesteps\nFine-Level Halo: [4,4,4]" at screen 0.2,0.8 font "Arial,14"
#set yrange [1:1000]

A = "small/output/scalingData"
B = "medium/output/scalingData"
C = "large/output/scalingData"

#D = "small/output/commData"
#E = "medium/output/commData"
#F = "large/output/commData"


# generate the x_curvefit
f1(x) = a1*x**b1               # define the function to be fit
a1 = 0.1; b1 = -1 
FIT_LIMIT=1e-6
fit f1(x) A using 2:5 via a1


# generate the x_curvefit
f2(x) = a2*x**b2               # define the function to be fit
a2 = 0.1; b2 = -1 
FIT_LIMIT=1e-6
fit f2(x) B using 2:5 via a2


# generate the x_curvefit
f3(x) = a3*x**b3               # define the function to be fit
a3 = 0.1; b3 = -1 
FIT_LIMIT=1e-6
fit f3(x) C using 2:5 via a3


plot A    using 2:5 with linespoints      ls 7   ps 1   lt rgb "green" lw 2 t 'L-1: 128^3, L-0: 32^3  ',\
     B    using 2:5 with linespoints      ls 7   ps 1   lt rgb "red"   lw 2 t 'L-1: 256^3, L-0: 64^3  ',\
     C    using 2:5 with linespoints      ls 7   ps 1   lt rgb "blue"  lw 2 t 'L-1: 512^3, L-0: 128^3',\
     ((x>=128) && (x<=4096))    ? f1(x) : (1/0)  ls 1 lt rgb "grey" lw 2 t 'Ideal',\
     ((x>=1024) && (x<=32768))  ? f2(x) : (1/0)  ls 1 lt rgb "grey" lw 2 t '', \
     ((x>=8192) && (x<=262144)) ? f3(x) : (1/0)  ls 1 lt rgb "grey" lw 2 t ''
     
#     D    using 2:3 with linespoints      ls 6     lt rgb "green" lw 1 t 'MPI Communication',\
#     E    using 2:3 with linespoints      ls 6     lt rgb "red"   lw 1 t 'MPI Communication',\
#     F    using 2:3 with linespoints      ls 6     lt rgb "blue"  lw 1 t 'MPI Communication'


   


#pause -1


!ps2pdf  -dEPSCrop scaling-2L-adaptive-rmcrt_DO.ps

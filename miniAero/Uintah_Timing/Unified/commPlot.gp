
#set term x11

# uncomment below for post script output
set terminal postscript landscape solid "Times-Roman" 12 size 10in,8in
set output "miniAeroScalingUnified_RK4.ps"
set style line 1  lt 1 lw 2
 
set autoscale
set logscale x
set logscale y
set grid xtics ytics

#set xtics ( "12" 12, "24" 24, "48" 48, "96" 96, "192" 192, "384" 384, "768" 768, "1536" 1535, "3072" 3072, "16384" 16384, "32768" 32768, "65536" 65536 )
set xtics ( "32" 32, "64" 64, "128" 128, "256" 256, "512" 512, "1024" 1024, "2048" 2048, "4096" 4096, "8192" 8192, "16K" 16384, "32K" 32768, "65K" 65536, "131K" 131072 )

set title "Uintah:Mini-Aero\n3D Riemann Problem, Single Level, OLCF-Titan System"

set ylabel "Average Time per Timestep [s]"
set label "Runge Kutta: 4\nViscous terms enabled\nUnified Scheduler\nAveraged over 23 timesteps\nOne patch per core" at screen 0.15,0.8

S="small"
M="medium"
L="large"
scale="/output/scalingData"
task="/output/aveComponentTimes"

set multiplot

# generate the x_curvefit
f1(x) = a1*x**b1               # define the function to be fit
a1 = 0.1; b1 = -1 
FIT_LIMIT=1e-6
fit f1(x) S.scale using 2:5 via a1

# generate the x_curvefit
f2(x) = a2*x**b2               # define the function to be fit
a2 = 0.1; b2 = -1 
FIT_LIMIT=1e-6
fit f2(x) M.task using 2:3 via a2



#__________________________________ TOP
set size   1.0, 0.333       
set origin 0.0, 0.666
#set yrange [0.01:20]

plot  S.scale    using 2:5 with linespoints lw 2 lt 8  lc 1 pointsize 1.0 t '16.1 M Cells (256^3) ',\
      M.scale    using 2:5 with linespoints lw 2 lt 8  lc 3 pointsize 1.0 t '134.2 M Cells (512^3) ',\
      L.scale    using 2:5 with linespoints lw 2 lt 8  lc 2 pointsize 1.0 t '1.0 B Cells (1024^3) ',\
      f1(x) ls 1 lt rgb "grey"  lw 2 t 'Ideal'
#__________________________________middle   
set origin 0.0,0.33
set title ""
set ylabel "Time [sec]"
set label "Task Execution Time" at screen 0.7,0.55

plot S.task    using 2:3 with linespoints lw 2 lt 8  lc 1 pointsize 1.0 t '',\
     M.task    using 2:3 with linespoints lw 2 lt 8  lc 3 pointsize 1.0 t '',\
     L.task    using 2:3 with linespoints lw 2 lt 8  lc 2 pointsize 1.0 t '',\
     f2(x) ls 1 lt rgb "grey"  lw 2 t 'Ideal'
     
#__________________________________bottom  
set origin 0.0,0.0
set title ""
set xlabel "Cores"
set ylabel "Time [sec]"
set autoscale
set label "Communication Wait Time" at screen 0.7,0.25

plot S.task    using 2:5 with linespoints lw 2 lt 8  lc 1 pointsize 1.0 t '',\
     M.task    using 2:5 with linespoints lw 2 lt 8  lc 3 pointsize 1.0 t '',\
     L.task    using 2:5 with linespoints lw 2 lt 8  lc 2 pointsize 1.0 t ''
set nomultiplot

!ps2pdf miniAeroScalingUnified_RK4.ps
!pdfcrop miniAeroScalingUnified_RK4.pdf

pause -1

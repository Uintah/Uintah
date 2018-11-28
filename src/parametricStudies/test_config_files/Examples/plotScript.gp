
# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "orderAccuracy.ps"
set style line 1  lt 1 lw 0.6 lc 1
set style line 2  lt 1 lw 0.6 lc 2
set style line 3  lt 1 lw 0.6 lc 3
 
set autoscale
set logscale x
set logscale y
set grid xtics ytics

#title
#xlabel
#ylabel

# generate the x_curvefit
f1(x) = a1*x**b1               # define the function to be fit
a1 = 0.1; 
b1 = -0.5 
fit [4:512] f1(x) 'L2norm.dat' using 1:2 via a1, b1

# generate the y_curvefit
f2(x) = a2*x**b2                # define the function to be fit
a2 = 0.1 
b2 = -0.5 
fit [4:512] f2(x) 'L2norm.dat' using 1:3 via a2, b2

# generate the z_curvefit
f3(x) = a3*x**b3                # define the function to be fit
a3 = 0.1 
b3 = -0.5
FIT_LIMIT=1e-8  
fit [4:512] f3(x) 'L2norm.dat' using 1:4 via a3, b3


set label 'x_Error = a * (#Rays)^b' at screen 0.2,0.4
set label sprintf( 'a = %2.3g',a1 ) at screen 0.3,0.375
set label sprintf( 'b = %2.3g',b1 ) at screen 0.3,0.35

set label 'y_Error = a * (#Rays)^b' at screen 0.2,0.3
set label sprintf( 'a = %2.3g',a2 ) at screen 0.3,0.275
set label sprintf( 'b = %2.3g',b2 ) at screen 0.3,0.25

set label 'z_Error = a * (#Rays)^b' at screen 0.2,0.2
set label sprintf( 'a = %2.3g',a3 ) at screen 0.3,0.175
set label sprintf( 'b = %2.3g',b3 ) at screen 0.3,0.15

set yrange [0.001:0.2]

plot 'L2norm.dat' using 1:2 t 'X Error' with points, \
     'L2norm.dat' using 1:3 t 'Y Error' with points, \
     'L2norm.dat' using 1:4 t 'Z Error' with points, \
     f1(x) ls 1 t 'x-curve fit', \
     f2(x) ls 2 t 'y-curve fit', \
     f3(x) ls 3 t 'z-curve fit'

!ps2pdf orderAccuracy.ps

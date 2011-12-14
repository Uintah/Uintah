set term png
set output "orderAccuracy.png"

# uncomment below for post script output
#set terminal postscript color solid "Times-Roman" 14
#set output "orderAccuracy.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3
#set style line 3  lt 1 lw 0.6 lc 5
 
set autoscale
set logscale x
set logscale y
set grid xtics ytics

#title
#xlabel
#ylabel

# generate the x_curvefit
f1(x) = a1*x**b1               # define the function to be fit
a1 = 0.1; b1 = -2.2 
FIT_LIMIT=1e-6
fit f1(x) 'L2norm.dat' using 1:2 via a1, b1

# generate the y_curvefit
f2(x) = a2*x**b2                # define the function to be fit
a2 = 0.1; b2 = -2.2      
FIT_LIMIT=1e-6  
fit f2(x) 'L2norm.dat' using 1:3 via a2, b2

# generate the z_curvefit
f3(x) = a3*x**b3                # define the function to be fit
a3 = 0.1; b3 = -2.2      
FIT_LIMIT=1e-6  
fit f3(x) 'L2norm.dat' using 1:4 via a3, b3

# ideal
f4(x) = a4*x**-1.0                # define the function to be fit
a4 = 0.1      
FIT_LIMIT=1e-6  
fit f4(x) 'L2norm.dat' using 1:4 via a4

set label 'Ideal = a * (#Grid Cells)^-1' at screen 0.2,0.475
set label 'a = %3.5g',a1      at screen 0.2,0.45

set label 'x_Error = a * (#Grid Cells)^b' at screen 0.2,0.4
set label 'a = %3.5g',a1      at screen 0.2,0.375
set label 'b = %3.5g',b1      at screen 0.2,0.35

set label 'y_Error = a * (#Grid Cells)^b' at screen 0.2,0.3
set label 'a = %3.5g',a2      at screen 0.2,0.275
set label 'b = %3.5g',b2      at screen 0.2,0.25

set label 'z_Error = a * (#Grid Cells)^b' at screen 0.2,0.2
set label 'a = %3.5g',a3      at screen 0.2,0.175
set label 'b = %3.5g',b3      at screen 0.2,0.15


plot 'L2norm.dat' using 1:2 t 'X Error' with points, \
     'L2norm.dat' using 1:3 t 'Y Error' with points, \
     'L2norm.dat' using 1:4 t 'Z Error' with points, \
     f1(x) t 'x-curve fit', \
     f2(x) t 'y-curve fit', \
     f3(x) t 'z-curve fit', \
     f4(x) t 'Ideal'


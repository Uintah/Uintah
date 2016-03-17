set term png 
set output "orderAccuracy.png"

# uncomment below for post script output
#set terminal postscript color solid "Times-Roman" 14
#set output "orderAccuracy.ps"

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

set label 'x_Error = a * (Spatial Resolution)^b' at screen 0.2,0.4
set label 'a = %3.5g',a1      at screen 0.2,0.375
set label 'b = %3.5g',b1      at screen 0.2,0.35

set label 'y_Error = a * (Spatial Resolution)^b' at screen 0.2,0.3
set label 'a = %3.5g',a2      at screen 0.2,0.275
set label 'b = %3.5g',b2      at screen 0.2,0.25



plot 'L2norm.dat' using 1:2 t 'uVelError' with linespoints, \
     'L2norm.dat' using 1:3 t 'vVelError' with linespoints, \
     f1(x) t 'x-curve fit', f2(x) t 'y-curve fit'


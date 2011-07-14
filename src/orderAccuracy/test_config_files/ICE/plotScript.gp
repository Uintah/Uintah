
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


# generate the curvefit
f1(x) = a1*x**b1                # define the function to be fit
a1 = 0.1; b1 = 0.01;            # initial guess for a1 and b1
fit f1(x) 'L2norm.dat' using 1:2 via a1, b1

set label 'Error = a * (Spatial Resolution)^b' at screen 0.2,0.4
set label 'a = %3.5g',a1      at screen 0.2,0.375
set label 'b = %3.5g',b1      at screen 0.2,0.35

plot 'L2norm.dat' using 1:2 t 'Current test' with linespoints,\
      f1(x) title "curve fit'


set term png 
set autoscale
set logscale x
set logscale y
set grid xtics ytics

set title "ICE:Riemann Problem (test 1) X dir"
set xlabel "Resolution"
set ylabel "Error"
set output "orderAccuracy.png"

# generate the curvefit
f1(x) = a1*x**b1                # define the function to be fit
f2(x) = a2*x**b2
f3(x) = a3*x**b3
f4(x) = a4*x**b4

a1 = 0.1; b1 = 0.01;            # initial guess
a2 = 0.1; b2 = 0.01;
a3 = 0.1; b2 = 0.01;
a4 = 0.1; b4 = 0.01;

fit f1(x) 'L2norm.dat' using 1:2 via a1, b1
fit f2(x) 'L2norm.dat' using 1:3 via a2, b2
fit f3(x) 'L2norm.dat' using 1:4 via a3, b3
fit f4(x) 'L2norm.dat' using 1:5 via a4, b4

set label 'Error = a * (Spatial Resolution)^ b' at screen 0.2,0.42

set label 'b'                           at screen 0.2,0.38
set label '___________________________' at screen 0.2,0.36
set label 'density     = %3.5g',b1      at screen 0.2,0.34
set label 'velocity    = %3.5g',b2      at screen 0.2,0.30
set label 'pressure    = %3.5g',b3      at screen 0.2,0.26
set label 'Temperature = %3.5g',b4      at screen 0.2,0.22

plot 'L2norm.dat' using 1:2 t 'Density' with linespoints,\
      f1(x) title "",\
      'L2norm.dat' using 1:3 t 'Velocity' with linespoints,\
      f2(x) title "",\
      'L2norm.dat' using 1:4 t 'Pressure' with linespoints,\
      f3(x) title "",\
      'L2norm.dat' using 1:5 t 'Temperature' with linespoints,\
      f4(x) title ""

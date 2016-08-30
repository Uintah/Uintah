set term png 
set output "orderAccuracy.png"

# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME1e15.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set grid xtics ytics
set key left top

set title "Propagation Velocity vs Time"
set xlabel "Time (s)"
set ylabel "Instantaneous Velocity (m/s)"



plot  'ConvectiveBurning_ME1e15HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e15HE1e8' with points,\
      'ConvectiveBurning_ME1e15HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e15HE1e5' with points
 
# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME1e8.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set grid xtics ytics
set key left top

set title "Propagation Velocity vs Time"
set xlabel "Time (s)"
set ylabel "Instantaneous Velocity (m/s)"
     
plot  'ConvectiveBurning_ME1e8HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e8HE1e8' with points,\
      'ConvectiveBurning_ME1e8HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e8HE1e5' with points
      
# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME1e7.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set grid xtics ytics
set key left top

set title "Propagation Velocity vs Time"
set xlabel "Time (s)"
set ylabel "Instantaneous Velocity (m/s)"
      
plot  'ConvectiveBurning_ME1e7HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e7HE1e8' with points,\
      'ConvectiveBurning_ME1e7HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e7HE1e5' with points    
      
# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME1e5.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set grid xtics ytics
set key left top

set title "Propagation Velocity vs Time"
set xlabel "Time (s)"
set ylabel "Instantaneous Velocity (m/s)"
      
plot   'ConvectiveBurning_ME1e5HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e5HE1e8' with points,\
      'ConvectiveBurning_ME1e5HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e5HE1e5' with points

# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME1e3.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set grid xtics ytics
set key left top

set title "Propagation Velocity vs Time"
set xlabel "Time (s)"
set ylabel "Instantaneous Velocity (m/s)"

plot  'ConvectiveBurning_ME1e3HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e8' with points,\
      'ConvectiveBurning_ME1e3HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e5' with points

# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "HE1e8.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set grid xtics ytics
set key left top

set title "Propagation Velocity vs Time"
set xlabel "Time (s)"
set ylabel "Instantaneous Velocity (m/s)"

plot  'ConvectiveBurning_ME1e15HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e8' with points,\
      'ConvectiveBurning_ME1e8HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e5' with points,\
      'ConvectiveBurning_ME1e7HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e8' with points,\
      'ConvectiveBurning_ME1e5HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e5' with points,\
      'ConvectiveBurning_ME1e3HE1e8.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e5' with points
      
# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "HE1e5.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set grid xtics ytics
set key left top

set title "Propagation Velocity vs Time"
set xlabel "Time (s)"
set ylabel "Instantaneous Velocity (m/s)"

plot  'ConvectiveBurning_ME1e15HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e8' with points,\
      'ConvectiveBurning_ME1e8HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e5' with points,\
      'ConvectiveBurning_ME1e7HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e8' with points,\
      'ConvectiveBurning_ME1e5HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e5' with points,\
      'ConvectiveBurning_ME1e3HE1e5.uda.000/vel.dat'   using 1:2 t 'ME1e3HE1e5' with points

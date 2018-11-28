set terminal png font "Times-Roman,11" size 1024,768
set output "Error.png"

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

plot 'L2norm.dat' using 1:2 t 'X Error' with points, \
     'L2norm.dat' using 1:3 t 'Y Error' with points, \
     'L2norm.dat' using 1:4 t 'Z Error' with points


set term png
set output "RR_error.png"

# uncomment below for post script output
#set terminal postscript color solid "Times-Roman" 14
#set output "RR_error.ps"
 
set autoscale
set grid xtics ytics

#title
#xlabel
#ylabel

set xtics ("1" 1, "2" 2, "4" 4, "8" 8, "16" 16)
set label "Resolution:\nCoarse level: 41,41,41 \nFine level:   RR * (41,41,41)" at screen 0.2,0.4
set label "Div Q computed on coarse level" at screen 0.2,0.2
set key at screen 0.35,0.175

plot 'L2norm.dat' using 1:2 t 'X Error' with points, \
     'L2norm.dat' using 1:3 t 'Y Error' with points, \
     'L2norm.dat' using 1:4 t 'Z Error' with points



# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "Error.ps"
 
set autoscale
set grid xtics ytics

set label "Burns & Christon Benchmark" screen 0.2, 0,8

#title
#xlabel
#ylabel

stats 'L2norm.dat' using 3 name "A"
set y2range[A_mean - A_stddev:A_mean + A_stddev]

set pointsize 2
plot 'L2norm.dat' using 2:xticlabels(1) t 'X Error' with points pt 7, \
     'L2norm.dat' using 3:xticlabels(1) t 'Y Error' with points pt 9, \
     'L2norm.dat' using 4:xticlabels(1) t 'Z Error' with points pt 13

!ps2pdf Error.ps

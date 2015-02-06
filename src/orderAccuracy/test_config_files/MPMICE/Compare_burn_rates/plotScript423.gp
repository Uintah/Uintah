set term png 
set output "orderAccuracy.png"

# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ResolutionAt423K.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure At 473K"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'Exp423.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_Resolution0.25mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution0.25mm' with points,\
      'DeterminingBurnRate_Resolution1mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution1mm' with points,\
      'DeterminingBurnRate_Resolution4mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution4mm' with points,\
      'DeterminingBurnRate_Resolution8mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution8mm' with points,\
      'DeterminingBurnRate_Resolution10mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution10mm' with points,\
      'DeterminingBurnRate_Resolution14mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution14mm' with points
      

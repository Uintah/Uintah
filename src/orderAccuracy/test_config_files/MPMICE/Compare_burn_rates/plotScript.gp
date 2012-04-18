set term png 
set output "orderAccuracy.png"

# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "VolFracAt298K.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure At 298K"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'ExpData.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_VolumeFraction1.uda.000/AP_mbr.dat'   using 1:2 t 'VolumeFraction1' with points,\
      'DeterminingBurnRate_VolumeFraction0.9.uda.000/AP_mbr.dat'   using 1:2 t 'VolumeFraction0.9' with points,\
      'DeterminingBurnRate_VolumeFraction0.8.uda.000/AP_mbr.dat'   using 1:2 t 'VolumeFraction0.8' with points,\
      'DeterminingBurnRate_VolumeFraction0.7.uda.000/AP_mbr.dat'   using 1:2 t 'VolumeFraction0.7' with points,\
      'DeterminingBurnRate_VolumeFraction0.6.uda.000/AP_mbr.dat'   using 1:2 t 'VolumeFraction0.6' with points
      

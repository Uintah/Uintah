set term png 
set output "orderAccuracy.png"

# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"



plot  'ExpData.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_ME3e8.uda.000/AP_mbr.dat'   using 1:2 t 'ME3e8' with points,\
      'DeterminingBurnRate_ME4e8.uda.000/AP_mbr.dat'   using 1:2 t 'ME4e8' with points,\
       'DeterminingBurnRate_ME5e8.uda.000/AP_mbr.dat'   using 1:2 t 'ME5e8' with points,\
      'DeterminingBurnRate_ME6e8.uda.000/AP_mbr.dat'   using 1:2 t 'ME6e8' with points,\
       'DeterminingBurnRate_ME7e8.uda.000/AP_mbr.dat'   using 1:2 t 'ME7e8' with points,\
      'DeterminingBurnRate_ME8e8.uda.000/AP_mbr.dat'   using 1:2 t 'ME8e8' with points,\
       'DeterminingBurnRate_ME9e8.uda.000/AP_mbr.dat'   using 1:2 t 'ME9e8' with points,\
      'DeterminingBurnRate_ME1e9.uda.000/AP_mbr.dat' using 1:2 t 'MEME1e9' with points


#Plot Next Momentum Exchange
      
 # uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME3e8.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'ExpData.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_ME3e8HE1e5.uda.000/AP_mbr.dat'   using 1:2 t 'ME3e8HE1e5' with points,\
      'DeterminingBurnRate_ME3e8HE3e5.uda.000/AP_mbr.dat'   using 1:2 t 'ME3e8HE3e5' with points,\
      'DeterminingBurnRate_ME3e8HE0.8e5.uda.000/AP_mbr.dat'   using 1:2 t 'ME3e8HE0.8e5' with points
      
      
#Plot Next Momentum Exchange
      
 # uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ME0.8e8.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'ExpData.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_ME0.8e8HE1e5.uda.000/AP_mbr.dat'   using 1:2 t 'ME0.8e8HE1e5' with points,\
      'DeterminingBurnRate_ME0.8e8HE3e5.uda.000/AP_mbr.dat'   using 1:2 t 'ME0.8e8HE3e5' with points,\
      'DeterminingBurnRate_ME0.8e8H0.8e5.uda.000/AP_mbr.dat'   using 1:2 t 'ME0.8e8H0.8e5' with points

      
# uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "Resolution.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'ExpData.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_Resolution1mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution1mm' with points,\
      'DeterminingBurnRate_Resolution1.33mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution1.33mm' with points,\
      'DeterminingBurnRate_Resolution2mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution2mm' with points,\
      'DeterminingBurnRate_Resolution0.5mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution0.5mm' with points,\
      'DeterminingBurnRate_Resolution0.25mm.uda.000/AP_mbr.dat'   using 1:2 t 'Resolution0.25mm' with points
      
#Plot Next Momentum Exchange
      
 # uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "ParticlesPerCell.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'ExpData.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_8ParticlesPerCell.uda.000/AP_mbr.dat'   using 1:2 t '8ParticlesPerCell' with points,\
      'DeterminingBurnRate_27ParticlesPerCell.uda.000/AP_mbr.dat'   using 1:2 t '27ParticlesPerCell' with points,\
      'DeterminingBurnRate_1ParticlePerCell.uda.000/AP_mbr.dat'   using 1:2 t '1ParticlePerCell' with points

#Plot Next Momentum Exchange
      
 # uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "EOS.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'ExpData.dat' using 1:2 t 'Experimental Data' with points,\
      'DeterminingBurnRate_JWLEOS.uda.000/AP_mbr.dat'   using 1:2 t 'JWLEOS' with points,\
      'DeterminingBurnRate_ModifiedEOS.uda.000/AP_mbr.dat'   using 1:2 t 'ModifiedEOS' with points

 # uncomment below for post script output
set terminal postscript color solid "Times-Roman" 14
set output "HighTemp.ps"
#set style line 1  lt 1 lw 0.6 lc 1
#set style line 2  lt 1 lw 0.6 lc 3

set autoscale
set logscale x
set logscale y
set grid xtics ytics
set key left top
set xrange [100000:1e8]

set title "Burn Rate vs Pressure"
set xlabel "Pressure (Pa)"
set ylabel "Burn Rate (m/s)"
     
      
plot  'Exp423.dat' using 1:2 t 'Experimental Data' with points,\
 'DeterminingBurnRate_Temp423Res1mm.uda.000/AP_mbr.dat'   using 1:2 t 'Temp423Res1mm' with points,\
 'DeterminingBurnRate_Temp423Res0.5mm.uda.000/AP_mbr.dat'   using 1:2 t 'Temp423Res0.5mm' with points,\
 'DeterminingBurnRate_Temp423Res0.25mm.uda.000/AP_mbr.dat'   using 1:2 t 'Temp423Res0.25mm' with points
      

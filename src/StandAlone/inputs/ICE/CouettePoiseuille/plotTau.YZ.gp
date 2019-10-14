set grid xtics ytics
set pointsize 1


#set terminal x11 1
set terminal  postscript color "Times-Roman" 10
set output "Tau_YZ.ps"

tau_Z = 'out.tau_Z_FC'
tau_Y = 'out.tau_Y_FC'

set title "Tau_{ZY} vs Tau_{YZ} \n Poiseuille Flow"
set multiplot
set size 1,0.51
set origin 0.0,0.5
set autoscale

#__________________________________
set xlabel "Z location"
set ylabel "|Shear stress|"

set xrange [-0.055: +0.055]
plot   tau_Z      using 3:(abs($5))            t 'tau_{ZY}'  w linespoints,\
       tau_Y      using 3:(abs($6))            t 'tau_{YZ}'  w linespoints,


set title
set xlabel
set ylabel
set size   0.33,0.51

set origin 0.0,0.0
set xrange [-0.055: -0.04]
plot   tau_Z      using 3:(abs($5))            t 'tau_{ZY}'  w linespoints,\
       tau_Y      using 3:(abs($6))            t 'tau_{YZ}'  w linespoints,

set origin 0.33,0.0
set xrange [-0.004: 0.004]
plot   tau_Z      using 3:(abs($5))            t 'tau_{ZY}'  w linespoints,\
       tau_Y      using 3:(abs($6))            t 'tau_{YZ}'  w linespoints,


set origin 0.66,0.0
set xrange [0.04: 0.055]
plot   tau_Z      using 3:(abs($5))            t 'tau_{ZY}'  w linespoints,\
       tau_Y      using 3:(abs($6))            t 'tau_{YZ}'  w linespoints,
       
!ps2pdf Tau_YZ.ps
#pause -1

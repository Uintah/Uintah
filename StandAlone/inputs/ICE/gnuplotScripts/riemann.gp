#__________________________________
#   Move data to a central location
#__________________________________
#!../inputs/ICE/gnuplotScripts/combinePatches
!cp ../inputs/ICE/gnuplotScripts/riemann.dat /tmp/exactSolution
!cp BOT_explicit_Pressure/79/L-0/patch_0/Press_CC /tmp/.
!cp BOT_Advection_after_BC/79/L-0/patch_0/Mat_0/Temp_CC /tmp/.
!cp BOT_Advection_after_BC/79/L-0/patch_0/Mat_0/X_vel_CC /tmp/.
!cp BOT_Advection_after_BC/79/L-0/patch_0/Mat_0/rho_CC /tmp/.
!echo "Done with moving data"

set ytics
set xtics
set mxtics
set mytics
#set grid mxtics ytics
set grid xtics ytics
set pointsize 0.5
set title "ICE, Second Order 100 cells"

#__________________________________
#   Pressure
#__________________________________
set autoscale
set terminal x11 1
#set terminal postscript "Times-Roman" 9
#set output "test1Comparison.ps"
set multiplot
set size 0.51,0.51  
set origin 0.0,0.0

set ylabel "mean pressure"
set y2tics
set xrange [0:1]
plot  '/tmp/exactSolution'    using 1:4 t 'exact' with lines, \
      '/tmp/Press_CC'         using 1:2 t ''

#__________________________________
#   Temp_CC
#__________________________________
set origin 0.5,0.0

set ylabel "Temp_CC"
plot  '/tmp/exactSolution'    using 1:5 t 'exact' with lines, \
      '/tmp/Temp_CC'          using 1:2 t ''

#__________________________________
#  uvel_CC
#__________________________________
set origin 0.0,0.5

set ylabel "uvel_CC"
plot  '/tmp/exactSolution'    using 1:3 t 'exact'  with lines, \
      '/tmp/X_vel_CC'         using 1:2 t ''
#__________________________________
#   rho_CC
#__________________________________
set origin 0.5,0.5

set ylabel "rho_CC"
plot  '/tmp/exactSolution'   using 1:2 t 'exact'  with lines, \
      '/tmp/rho_CC'       using 1:2 t ''
      
set nomultiplot 
set size 1.0, 1.0
set origin 0.0, 0.0
pause -1 "Hit return to continue"
#__________________________________
#  cleanup the mess
#__________________________________
!/bin/rm /tmp/rho_CC
!/bin/rm /tmp/X_vel_CC
!/bin/rm /tmp/Temp_CC
!/bin/rm /tmp/Press_CC
!/bin/rm /tmp/exactSolution

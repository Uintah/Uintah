#__________________________________
#  cleanup the mess
#__________________________________
!/bin/rm -f /tmp/rho_CC
!/bin/rm -f /tmp/X_vel_CC
!/bin/rm -f /tmp/Temp_CC
!/bin/rm -f /tmp/Press_CC
!/bin/rm -f /tmp/rho_CC_L1
!/bin/rm -f /tmp/X_vel_CC_L1
!/bin/rm -f /tmp/Temp_CC_L1
!/bin/rm -f /tmp/Press_CC_L1
!/bin/rm -f /tmp/rho_CC_L2
!/bin/rm -f /tmp/X_vel_CC_L2
!/bin/rm -f /tmp/Temp_CC_L2
!/bin/rm -f /tmp/Press_CC_L2
!/bin/rm -f /tmp/exactSolution

#__________________________________
#   Move data to a central location
#__________________________________
!../scripts/ICE/combinePatches
!cp ../scripts/ICE/riemann.dat /tmp/exactSolution
!cp BOT_explicit_Pressure/971/L-0/patch_combined/Press_CC /tmp/.
!cp BOT_Advection_after_BC/971/L-0/patch_combined/Mat_0/Temp_CC /tmp/.
!cp BOT_Advection_after_BC/971/L-0/patch_combined/Mat_0/X_vel_CC /tmp/.
!cp BOT_Advection_after_BC/971/L-0/patch_combined/Mat_0/rho_CC /tmp/.

!cp BOT_explicit_Pressure/971/L-1/patch_combined/Press_CC /tmp/Press_CC_L1
!cp BOT_Advection_after_BC/971/L-1/patch_combined/Mat_0/Temp_CC /tmp/Temp_CC_L1
!cp BOT_Advection_after_BC/971/L-1/patch_combined/Mat_0/X_vel_CC /tmp/X_vel_CC_L1 
!cp BOT_Advection_after_BC/971/L-1/patch_combined/Mat_0/rho_CC /tmp/rho_CC_L1

!cp BOT_explicit_Pressure/971/L-2/patch_combined/Press_CC /tmp/Press_CC_L2
!cp BOT_Advection_after_BC/971/L-2/patch_combined/Mat_0/Temp_CC /tmp/Temp_CC_L2
!cp BOT_Advection_after_BC/971/L-2/patch_combined/Mat_0/X_vel_CC /tmp/X_vel_CC_L2 
!cp BOT_Advection_after_BC/971/L-2/patch_combined/Mat_0/rho_CC /tmp/rho_CC_L2
!echo "Done with moving data"

set ytics
set xtics
set mxtics
set mytics
#set grid mxtics ytics
set grid xtics ytics
set pointsize 0.5
set title "Shock Tube with 3 Levels"

#__________________________________
#   Pressure
#__________________________________
set autoscale
set terminal x11 1
#set terminal enhanced postscript color "Times-Roman" 9
#set output "goodness.ps"
set multiplot
set size 0.51,0.51  
set origin 0.0,0.0

set ylabel "mean pressure"
set y2tics
set xrange [0:1]
set key left bottom Left
plot  '/tmp/exactSolution'    using 1:4 t 'exact'        with lines, \
      '/tmp/Press_CC'         using 1:2 t 'coarse level' with lines,\
      '/tmp/Press_CC_L1'      using 1:2 t 'L-1'   with lines,\
      '/tmp/Press_CC_L2'      using 1:2 t 'L-2'   with lines

#__________________________________
#   Temp_CC
#__________________________________
set origin 0.5,0.0

set ylabel "Temp_CC"
plot  '/tmp/exactSolution'    using 1:5 t 'exact'        with lines, \
      '/tmp/Temp_CC'          using 1:2 t 'coarse level' with lines,\
      '/tmp/Temp_CC_L1'       using 1:2 t 'L-1'   with lines,\
      '/tmp/Temp_CC_L2'       using 1:2 t 'L-2'   with lines

#__________________________________
#  uvel_CC
#__________________________________
set origin 0.0,0.5

set ylabel "uvel_CC"
plot  '/tmp/exactSolution'    using 1:3 t 'exact'        with lines, \
      '/tmp/X_vel_CC'         using 1:2 t 'coarse level' with lines,\
      '/tmp/X_vel_CC_L1'      using 1:2 t 'L-1'   with lines,\
      '/tmp/X_vel_CC_L2'      using 1:2 t 'L-2'   with lines
#__________________________________
#   rho_CC
#__________________________________
set origin 0.5,0.5

set ylabel "rho_CC"
plot  '/tmp/exactSolution'   using 1:2 t 'exact'        with lines,\
      '/tmp/rho_CC'          using 1:2 t 'coarse level' with lines,\
      '/tmp/rho_CC_L1'       using 1:2 t 'L-1'   with lines,\
      '/tmp/rho_CC_L2'       using 1:2 t 'L-2'   with lines
      
set nomultiplot 
set size 1.0, 1.0
set origin 0.0, 0.0
pause -1 "Hit return to continue"

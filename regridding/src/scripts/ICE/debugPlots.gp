#__________________________________
#  cleanup the mess
#__________________________________
!/bin/rm /tmp/rho_CC
!/bin/rm /tmp/X_vel_CC
!/bin/rm /tmp/Y_vel_CC
!/bin/rm /tmp/Z_vel_CC
!/bin/rm /tmp/Temp_CC
!/bin/rm /tmp/press_CC
!/bin/rm /tmp/sp_vol_CC
!/bin/rm /tmp/sp_vol_L
!/bin/rm /tmp/uvel_FC
!/bin/rm /tmp/vvel_FC
!/bin/rm /tmp/wvel_FC
!/bin/rm /tmp/sumDelPress
!/bin/rm /tmp/mass_advected
#__________________________________
#   Move data to a central location
# <debug label = "switchDebug_advance_advect"/>
# <debug label = "switchDebug_vel_FC"/>
# <debug label = "switchDebug_computeDelP"/>
# <debug label = "switchDebug_vel_FC"/>
#__________________________________
!../inputs/ICE/gnuplotScripts/combinePatches

!cp BOT_computeDelP/957/L-0/patch_combined/press_CC /tmp/.
!cp BOT_computeDelP/957/L-0/patch_combined/delP_Dilatate /tmp/.
!cp BOT_computeVel_FC/957/L-0/patch_combined/Mat_0/uvel_FC /tmp/.
!cp BOT_computeVel_FC/957/L-0/patch_combined/Mat_0/vvel_FC /tmp/.
!cp BOT_computeVel_FC/957/L-0/patch_combined/Mat_0/wvel_FC /tmp/.

!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/Temp_CC /tmp/.
!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/X_vel_CC /tmp/.
!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/Y_vel_CC /tmp/.
!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/Z_vel_CC /tmp/.
!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/rho_CC    /tmp/.
!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/sp_vol_CC /tmp/.
!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/sp_vol_L  /tmp/.
!cp BOT_Advection_after_BC/957/L-0/patch_combined/Mat_0/mass_advected  /tmp/.
!echo "Done with moving data"

set ytics
set xtics
set mxtics
set mytics
#set grid mxtics ytics
set grid xtics ytics
set pointsize 1


#__________________________________
#   Pressure
#__________________________________
set autoscale
#set terminal x11 2
set terminal enhanced postscript "Times-Roman" 9
set output "primative.ps"
set multiplot
set size 0.51,0.51  
set origin 0.0,0.0
set title "After Advection"
set ylabel "mean pressure"
set y2tics
set xrange [0.6:0.8]

plot   '/tmp/press_CC'         using 1:2 t 'press'

#__________________________________
#   Temp_CC
#__________________________________
set origin 0.5,0.0

set ylabel "Temp_CC"
plot   '/tmp/Temp_CC'          using 1:2  t 'temp' w linespoints
#__________________________________
#  vel_CC
#__________________________________ 
set origin 0.0,0.5

set ylabel "vel_CC"
plot   '/tmp/Y_vel_CC'         using 1:2 t 'Yvel'
#__________________________________
#   rho_CC
#__________________________________
set origin 0.5,0.5

set ylabel "rho_CC"
plot   '/tmp/rho_CC'       using 1:2 t ''
      
set nomultiplot 
set size 1.0, 1.0
set origin 0.0, 0.0


#__________________________________
#   sp_vol_CC
#__________________________________
set title ""
set terminal x11 3
set multiplot
set size 0.51,0.51
set origin 0.0,0.0

set ylabel "sp_vol_CC"
plot   '/tmp/sp_vol_CC'       using 1:2 t 'sp_vol_CC'

#__________________________________
#   sp_vol_L
#__________________________________
set origin 0.0,0.5

set ylabel "sp_vol_L"
plot   '/tmp/sp_vol_L'          using 1:2 t ''
#__________________________________
#   delP
#__________________________________
set origin 0.5,0.0

set ylabel "delP"
plot   '/tmp/delP_Dilatate'          using 1:2 t ''
#__________________________________
#   vel_FC
#__________________________________
set origin 0.5,0.5

set ylabel "vel_FC"
plot   '/tmp/vvel_FC'          using 1:2 t 'vvel'       
            
set nomultiplot 
set size 1.0, 1.0
set origin 0.0, 0.0


#__________________________________
#   massAdvected
#__________________________________
set autoscale
set terminal x11 4
set ylabel "mass_advected"
set y2tics
plot   '/tmp/mass_advected'         using 1:2 t ''

#pause -1 "Hit return to continue"


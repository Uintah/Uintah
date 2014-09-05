#_________________________________________________________________________
#   06/15/04
#   This gnuplot script plots the total mass, momentum, kinetic energy and 
#   internal energy from the dat files.  You need to add
#
#      <save label="TotalMass"/>
#      <save label="KineticEnergy"/>
#      <save label="TotalIntEng"/>
#      <save label="CenterOfMassVelocity"/>
#   and
#      <debug label = "switchDebug_explicit_press"/> 
#
#   to the ups file.
#_________________________________________________________________________
#_________________________________
#  rip out "[" "]" from center of mass
!sed 's/\[//g' CenterOfMassVelocity.dat | sed 's/\]//g' >TotalMom.dat

set ytics
set xtics
set mxtics
set mytics
#set grid mxtics ytics
set grid xtics ytics
set pointsize 1

set autoscale
set terminal x11 1
#set terminal postscript "Times-Roman" 9
#set output "datPlots.ps"

#set xrange[0:0.02]
#__________________________________
#   TotalMass
#__________________________________
set multiplot
set size 0.51,0.51  
set origin 0.0,0.0
set ylabel "total mass"
set y2tics

plot   'TotalMass.dat'      using 1:2 t ''  w lines

#__________________________________
#   totalInternalEnergy
#__________________________________
set origin 0.5,0.0

set ylabel "Total Internal Energy"
plot   'TotalIntEng.dat'    using 1:2  t '' w lines

#__________________________________
#  KineticEnergy.dat
#__________________________________ 
set origin 0.0,0.5

set ylabel "KineticEnergy"
plot   'KineticEnergy.dat'  using 1:2 t ''  w lines

#__________________________________
#  totalMomentum.dat
#__________________________________ 
set origin 0.5,0.5

set ylabel "total Momentum"
plot   'TotalMom.dat'         using 1:2 t 'x'  w lines,\
       'TotalMom.dat'         using 1:3 t 'y'  w lines, \
       'TotalMom.dat'         using 1:4 t 'z'  w lines
       
set nomultiplot   
pause -1 "Hit return to continue"

exit       

#---------------------------------------------------------------
#    E X C H A N G E   E R R O R   optional
!sed 's/\[//g' mom_exch_error.dat | sed 's/\]//g' >mom_exch_error_clean.dat
#__________________________________
#   eng_exch_error
#__________________________________

set terminal x11 2
set multiplot
set size 1.0,0.51  
set origin 0.0,0.0
set ylabel "energy exchange error"
set y2tics

plot   'eng_exch_error.dat'         using 1:2 t ''  w line

#__________________________________
#   totalInternalEnergy
#__________________________________
set origin 0.0,0.5

set ylabel "momentum exchange error"
set yrange[-1e-20:1e-20]
plot   'mom_exch_error_clean.dat'         using 1:3 t 'x'  w lines,\
       'mom_exch_error_clean.dat'         using 1:4 t 'y'  w lines, \
       'mom_exch_error_clean.dat'         using 1:5 t 'z'  w lines

set nomultiplot 

#pause -1 "Hit return to continue"


#_________________________________________________________________________
#   This gnuplot script plots the total mass, momentum, kinetic energy and 
#   internal energy from the dat files.  You need to add
# 
#       <save label = "KineticEnergy"/>
#       <save label = "TotalMass"/>
#       <save label = "StrainEnergy"/>
#       <save label = "CenterOfMassPosition"/>
#       <save label = "TotalMomentum"/>
#   to the ups file.
#_________________________________________________________________________
#_________________________________
#  rip out "[" "]" from center of mass
!sed 's/\[//g' TotalMomentum.dat | sed 's/\]//g' >TotalMom.dat

set ytics
set xtics
set mxtics
set mytics
#set grid mxtics ytics
set grid xtics ytics
set pointsize 1

set autoscale
set terminal x11 1
#set terminal enhanced postscript "Times-Roman" 9
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
#   StrainEnergy
#__________________________________
set origin 0.5,0.0

set ylabel "Strain Energy"
plot   'StrainEnergy.dat'    using 1:2  t '' w lines

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

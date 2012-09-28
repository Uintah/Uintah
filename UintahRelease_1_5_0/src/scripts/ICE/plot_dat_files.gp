#_________________________________________________________________________
#   06/15/04
#   This gnuplot script plots the total mass, momentum, kinetic energy and 
#   internal energy from the dat files.  You need to add
#
#      <save label="TotalMass"/>
#      <save label="KineticEnergy"/>
#      <save label="TotalIntEng"/>
#      <save label="TotalMomentum"/> 
#
#   to the ups file.
#_________________________________________________________________________

#  rip out "[" "]" from center of mass
!sed 's/\[//g' TotalMomentum.dat | sed 's/\]//g' >TotalMom.dat

# compute the relative quantities
!awk 'NR==1 { init = $2}; NR>1 {printf("%16.15f %16.15f %16.15f\n", $1, $2, 1.0 - $2/init) }' TotalMass.dat > tmp
!mv tmp TotalMass2.dat

!awk 'NR==1 { init = $2}; NR>1 {printf("%16.15f %16.15f %16.15f\n", $1, $2, 1.0 - $2/init) }' TotalIntEng.dat > tmp
!mv tmp TotalIntEng2.dat


!awk 'NR==1 { init = $2}; NR>1 {printf("%16.15f %16.15f %16.15f\n", $1, $2, 1.0 - $2/init) }' KineticEnergy.dat > tmp
!mv tmp KineticEnergy2.dat

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
set ylabel "total mass"            textcolor lt 1
set y2label "Relative Difference"  textcolor lt 2
set y2tics

plot   'TotalMass2.dat'      using 1:2            t ''  w lines,\
       'TotalMass2.dat'      using 1:3  axes x1y2 t 'relative' w lines

#__________________________________
#   totalInternalEnergy
#__________________________________
set origin 0.5,0.0

set ylabel "Total Internal Energy" textcolor lt 1
set y2label "Relative Difference"  textcolor lt 2
plot   'TotalIntEng2.dat'    using 1:2            t '' w lines,\
       'TotalIntEng2.dat'    using 1:3  axes x1y2 t 'relative' w lines

#__________________________________
#  KineticEnergy.dat
#__________________________________ 
set origin 0.0,0.5

set ylabel "Kinetic Energy" textcolor lt 1
set y2label "Relative Difference"  textcolor lt 2
plot   'KineticEnergy2.dat'  using 1:2            t ''  w lines,\
       'KineticEnergy2.dat'  using 1:3  axes x1y2 t 'relative' w lines

#__________________________________
#  totalMomentum.dat
#__________________________________ 
set origin 0.5,0.5

set ylabel "total Momentum"
set y2label ""
plot   'TotalMom.dat'         using 1:2 t 'x'  w lines,\
       'TotalMom.dat'         using 1:3 t 'y'  w lines, \
       'TotalMom.dat'         using 1:4 t 'z'  w lines
       
set nomultiplot   
pause -1 "Hit return to continue"

# cleanup
!/bin/rm TotalMass2.dat TotalIntEng2.dat KineticEnergy2.dat 

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


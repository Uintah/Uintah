#set terminal X11
set terminal postscript color enhanced landscape "Times-Roman" 10
set output "plotEnergy.2.ps

set style line 1  lt 1 lw 2 lc -1
set style line 2  lt 1 lw 2 lc 1
set style line 3  lt 1 lw 2 lc 2

set grid xtics ytics 
set y2tics

set title "Resolution Study:Steel Ball Impacting a Steel Container \n \
           Constitutive Model: comp neo hook damage, yield stress 792.0e6, hardening modulus: 8e9  \n \
           Ball initial velocity 200m/s"
           
set ylabel "Energy"
set xlabel 'Physical Time[sec]'

set label " Failure Model:\n Failure stress: 6e9\n Std: 10%\n Distribution: gauss" at screen 0.5,0.5

uda = "441_200ms.uda.002"

# create one file with all of the data in it
syscall=sprintf("paste %s/KineticEnergy.dat %s/StrainEnergy.dat > Energy.dat",uda,uda)
system syscall

#set xrange [0.000:0.0003]
#set yrange [0.0:30]

plot 'Energy.dat'   using 1:($2 + $4)  t  'Total Energy'   with l ls 1,\
     'Energy.dat'   using 1:2          t  'Kinetic Energy' with l ls 2,\
     'Energy.dat'   using 1:4          t  'Strain Energy'  with l ls 3
     
pause -1 "Hit return to exit"

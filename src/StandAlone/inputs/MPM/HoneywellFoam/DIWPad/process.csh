cp particleExtract/L-0/21392098230009856 particleWHeader.txt
#rm -rf particleExtract
tail -n +2 particleWHeader.txt > particle.txt
paste -d ' ' particle.txt BndyForce_zminus.dat > allofit
gawk '{print $1,$7,$11}' allofit > time_disp_force.dat
rm -f allofit

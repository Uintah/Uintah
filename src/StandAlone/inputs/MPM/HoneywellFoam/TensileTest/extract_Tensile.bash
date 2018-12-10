#!/bin/bash
## The numbers after particleExtract/L-0 will change as the resolution of the simulation changes.
tail -n +3 particleExtract/L-0/4296278017 | awk '{print $1, $2, $3, $4}' > particle1.txt
tail -n +3 particleExtract/L-0/4297326593 | awk '{print $1, $2, $3, $4}' > particle2.txt
paste -d ' ' particle1.txt particle2.txt > particles.txt
rm -f particle1.txt
rm -f particle2.txt
cat particles.txt | awk '{print $1,$2-$6,$3-$7,$4-$8}' > particle_offsets.txt
rm -f particles.txt
cat particle_offsets.txt | awk '{print $1,sqrt($2*$2+$3*$3+$4*$4)}' > particle_distances.txt
paste -d ' ' particle_distances.txt BndyForce_xminus.dat > total.txt
rm -f particle_offsets.txt
rm -f particle_distances.txt
awk '{printf "%14.8f %14.8f %14.8f \n", $1,    $2,    $4}' total.txt > time_dist_force.txt
rm -f total.txt

#!/bin/bash
##  The number after particleExtract/L-0 will change as the resolution of the system changes
tail -n +3 particleExtract/L-0/4295491585 | awk '{print $1, $2, $5 }' > particle_x.txt
paste -d ' ' particle_x.txt BndyForce_xminus.dat > total.txt
rm -f particle_x.txt
awk '{printf "%14.8f %14.8f %14.8f \n", $1,    $2,    $5}' total.txt > time_dist_force.txt
rm -f total.txt

#!/bin/bash
# First for material 0
m=0
istart=6
iend=$((istart+33))
k=$((iend-istart))
for ((i=istart; i<=iend; i=i+1))
do 
      k=$((i-istart))
      b="LS.${k}.${m}.txt"
      echo $b
      ../partextract -mat $i -partvar ls.MidToEndVector -include_position_output -timestep 194 . > $b
done

# Next for material 1
m=1
istart=40
#iend=75
iend=$((istart+35))
k=$((iend-istart))
for ((i=istart; i<=iend; i=i+1))
do 
      k=$((i-istart))
      b="LS.${k}.${m}.txt"
      echo $b
      ../partextract -mat $i -partvar ls.MidToEndVector -include_position_output -timestep 194 . > $b
done

# Next for material 2
m=2
istart=75
iend=$((istart+37))
k=$((iend-istart))
for ((i=istart; i<=iend; i=i+1))
do 
      k=$((i-istart))
      b="LS.${k}.${m}.txt"
      echo $b
      ../partextract -mat $i -partvar ls.MidToEndVector -include_position_output -timestep 194 . > $b
done

# Next for material 3
m=3
istart=112
iend=$((istart+29))
k=$((iend-istart))
for ((i=istart; i<=iend; i=i+1))
do 
      k=$((i-istart))
      b="LS.${k}.${m}.txt"
      echo $b
      ../partextract -mat $i -partvar ls.MidToEndVector -include_position_output -timestep 194 . > $b
done

# Next for material 4
m=4
istart=141
iend=$((istart+9))
k=$((iend-istart))
for ((i=istart; i<=iend; i=i+1))
do 
      k=$((i-istart))
      b="LS.${k}.${m}.txt"
      echo $b
      ../partextract -mat $i -partvar ls.MidToEndVector -include_position_output -timestep 194 . > $b
done

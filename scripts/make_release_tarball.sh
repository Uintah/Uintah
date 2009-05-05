#! /bin/sh

cd ..

for i in `find . -name "*.release"`; do 
    j=`echo $i | sed 's/.release//'`;  
    cp $i $j 
 
done




exit 0
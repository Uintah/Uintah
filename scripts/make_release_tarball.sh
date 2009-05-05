#! /bin/sh

cd ..

for i in `find . -name "*.release"`; do 
    j=`echo $i | sed 's/.release//'`;  
    cp $i $j 
 
done

cd ../doc

./runLatex

cd ..

tar -X exclude.txt --exclude-vcs -cvf uintah.tar doc src



exit 0
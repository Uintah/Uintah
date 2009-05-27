#! /bin/sh

#VERSION

VERSION="1.1.0-alpha"

# Start off in the src/scripts directory.

cd ../..

mkdir uintah-$VERSION

cp -a doc src uintah-$VERSION

cd uintah-$VERSION/src

for i in `find . -name "*.release"`; do 
    j=`echo $i | sed 's/.release//'`;  
    cp $i $j 
 
done

cd ../doc

./runLatex

cd ../..

tar -X src/exclude.txt --exclude-vcs -cvf uintah-$VERSION.tar uintah-$VERSION

gzip uintah-$VERSION.tar



exit 0

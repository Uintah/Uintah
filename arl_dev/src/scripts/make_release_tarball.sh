#! /bin/sh

#VERSION

VERSION="1.1.0-alpha"

while getopts v: opt

do 
  	case "$opt" in
	   v) VERSION="$OPTARG";;
        esac
done
shift `expr $OPTIND - 1`


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

mv DeveloperGuide/UintahAPI.pdf .
mv InstallationGuide/installation_guide.pdf .
mv UserGuide/user_guide.pdf .

rm -rf DeveloperGuide figures InstallationGuide UserGuide movies Other README runLatex

cd ../..

tar -X src/exclude.txt --exclude-vcs -cvf uintah-$VERSION.tar uintah-$VERSION

gzip uintah-$VERSION.tar



exit 0

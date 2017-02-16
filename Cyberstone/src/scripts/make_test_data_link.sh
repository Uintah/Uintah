#!/bin/sh

mkdir -p TestData
cd TestData/
for i in $1/*; 
	do ln -sf $i .; 
done

exit

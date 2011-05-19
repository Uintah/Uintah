#!/bin/sh

mkdir -p TestData
cd TestData/
for i in /usr/local/TestData/opt/*; 
	do ln -sf $i .; 
done

exit

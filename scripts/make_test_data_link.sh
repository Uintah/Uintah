#!/bin/sh

mkdir TestData
cd TestData/
for i in /usr/local/TestData/opt/*; 
	do ln -s $i .; 
done

exit

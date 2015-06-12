#! /bin/bash

rm -f errors; 

if [ $# -eq 1 ]; then
   ups_files=`find -L . -name "*.ups" | grep "/$1/" `
else
   ups_files=`find -L . -name "*.ups"`
fi


for i in ${ups_files}; 
	do echo $i > .output; 
	sus -validate $i >> .output 2>&1; 
	if [ $? -ne 0 ]; then 
		cat .output >> errors; 
	fi 
done 

if [ -e errors ]; then
	less errors
else
	echo "No errors"
fi

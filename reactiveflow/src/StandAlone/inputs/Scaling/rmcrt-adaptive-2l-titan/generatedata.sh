#!/bin/bash

ext=*.out;

for folder in {small,medium,large};
    do 
        cd $folder/output;
        ../../extractScalingData *.$ext;
        cd ../../;
    done


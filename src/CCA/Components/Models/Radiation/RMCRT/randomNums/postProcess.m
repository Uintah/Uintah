#! /usr/bin/env octave

clear all;
close all;
format short e;

data = load('randomNumbers.dat');
i           = data(:,1);
randCPU       = data(:,2);
randHostGPU   = data(:,3);
randDevGPU_M  = data(:,4);
randDevGPU_N  = data(:,5);

plot( i, randCPU,'+',      "markersize", 2,
      i, randHostGPU,'o',  "markersize", 2,
      i, randDevGPU_M,'x', "markersize", 2,
      i, randDevGPU_N,'*', "markersize", 2);
      
legend( {'CPU', 'hostCPU', 'GPU dblExc', 'GPU dblInc'}, "location", "eastoutside");
legend boxon;
pause

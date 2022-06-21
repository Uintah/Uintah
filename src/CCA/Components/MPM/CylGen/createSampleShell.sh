#! /bin/sh
echo 'Compiling code'
g++ -O3 -o cylgen cylgen.cc
g++ -O3 -o resizeCylinders resizeCylinders.cc
g++ -O3 -o makeGroups makeGroups.cc
g++ -O3 -o MakeLineSegmentsCircleVar MakeLineSegmentsCircleVar.cc
g++ -O3 -o MakeLineSegmentsPiston MakeLineSegmentsPiston.cc
echo 'Generating cylinders'
cylgen
echo 'Resizing cylinders'
resizeCylinders
mv Position_Radius.RS.txt Position_Radius.txt
echo 'Grouping cylinders'
makeGroups
echo 'Making directory for Line Segments'
mkdir LineSegments
echo 'Making Line Segments for cylinders'
MakeLineSegmentsCircleVar
echo 'Making Line Segments for Piston'
MakeLineSegmentsPiston

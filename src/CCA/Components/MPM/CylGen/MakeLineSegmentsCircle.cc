// This code generates line segments that surround a distribution of cylinders
// described in a file called PositionRadius.N.txt where "N" is the group
// number.

// To compile:

//  g++ -O3 -o MakeLineSegmentsCircle MakeLineSegmentsCircle.cc

// To run:

// >MakeLineSegmentsCircle

// This will create files called "LineSegments.N.txt"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace std;


int main(int argc, char** argv)
{

 int groupNum=0;
 while(1){
  // Open the files to get cylinder data
  ostringstream gnum;
  gnum << groupNum;
  groupNum++;
  string infileroot("Position_Radius.");
  string infileext(".txt");
  string infilename = infileroot + gnum.str() + infileext;
  ifstream circles(infilename.c_str());
  if(!circles){
    cerr << "Input file not opened, exiting" << endl;
    exit(1);
  }

  double xcen;
  double ycen;
  double radius;
  int numLSs=200;

  double theta0 = 0.0001;
  int circleNum = 0;
  while(circles >> xcen >> ycen >> radius){
    ostringstream fnum, LSnum;
    string filename;
    string fileroot("LineSegments.");
    string fileext(".txt");
    fnum << circleNum;
    //LSnum << numLSs;
    filename = fileroot +/* LSnum.str() + "."*/ + fnum.str() 
             + "." + gnum.str() + fileext;
    ofstream out(filename.c_str());

    circleNum++;
    double circumference = 2.*M_PI*radius;

    vector<double> xpos(numLSs), ypos(numLSs);

    for(int index=0;index<numLSs;index++){
      double theta = theta0 + ((double) index)*2.*M_PI/((double) numLSs);
      xpos[index] = radius*sin(theta)+xcen;
      ypos[index] = radius*cos(theta)+ycen;
    }

    for(int index=0;index<numLSs;index++){
      out << xpos[index]  << " " << ypos[index]  << endl;
    }
    theta0+=M_PI/((double) numLSs);
    out.close();
  }
  circles.close();
 }
}

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

 double topOfCylinders = 0.1-1.e-5;
 int groupNum=0;
 string filename;
 string XMLfilename;
 string directory("LineSegments/");
 string fileroot("LineSegmentsCircle.");
 string fileext(".txt");
 XMLfilename = directory + fileroot + "xml";
 ofstream XMLout(XMLfilename.c_str());
 if(!XMLout){
   cerr << "File " << XMLfilename << " can't be opened." << endl;
 }
 XMLout << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
 XMLout << "<Uintah_Include>" << endl;
 bool moreFilesToWorkOn = true;
 while(moreFilesToWorkOn){
   // Open the files to get cylinder data
   ostringstream gnum;
   gnum << groupNum;
   string infileroot("Position_Radius.");
   string infileext(".txt");
   string infilename = infileroot + gnum.str() + infileext;
   ifstream circles(infilename.c_str());
 
   if(circles){
 
     double xcen;
     double ycen;
     double radius;
     int numLSs=200;

     double theta0 = 0.0001;
     int circleNum = 0;

     while(circles >> xcen >> ycen >> radius){
       ostringstream fnum, LSnum;
       fnum << circleNum;
       //LSnum << numLSs;
       filename = directory + fileroot + fnum.str() + "." + gnum.str() +fileext;
       ofstream out(filename.c_str());
   
       circleNum++;
       double circumference = 2.*M_PI*radius;

       vector<double> xpos(numLSs), ypos(numLSs);

       for(int index=0;index<numLSs;index++){
         double theta = theta0 + ((double) index)*2.*M_PI/((double) numLSs);
         xpos[index] = radius*sin(theta)+xcen;
         ypos[index] = min(radius*cos(theta)+ycen,topOfCylinders);
       }

       for(int index=0;index<numLSs;index++){
         out << xpos[index]  << " " << ypos[index]  << endl;
       }
       theta0+=M_PI/((double) numLSs);
       out.close();
       XMLout << "  <LineSegment>" << endl;
       XMLout << "    <associated_material> " << groupNum << " </associated_material>" << endl;
       XMLout << "    <lineseg_filename>" << filename << " </lineseg_filename>" << endl;
       XMLout << "  </LineSegment>" << endl;
     }
     circles.close();
     groupNum++;
   } else {
     cerr << "Finished or can't find " << infilename  << "." << endl;
     moreFilesToWorkOn = false;
   }
  }
  XMLout << "</Uintah_Include>" << endl;
  XMLout.close();
}

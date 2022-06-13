// This code generates line segments that surround a box
// the coordinates of which are given at the top of main()

// To compile:

//  g++ -O3 -o MakeLineSegmentsPiston MakeLineSegmentsPiston.cc

// To run:

// >MakeLineSegmentsPiston

// This will create a file called "LineSegmentsPiston.txt"
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
  // Now the piston
  double xmin_block= 0.00+1.e-5;
  double xmax_block= 0.10-1.e-5;
  double ymin_block= 0.10+1.e-5;
  double ymax_block= 0.12-1.e-5;
  double dx_tr=0.0005;

#if 0
               <box label="piston">
                 <min>[ 0.0, 0.10,-1.0] </min>
                 <max>[ 0.1, 0.11, 1.0] </max>
               </box>
#endif

  ofstream outp("LineSegments/LineSegmentsPiston.txt");
  vector<double> xpos, ypos;

  // Left face
  double x=xmin_block;
  double y=ymin_block;
  while(y < ymax_block){
    xpos.push_back(x);
    ypos.push_back(y);
    y+=dx_tr;
  }

  // Top face
  x=xmin_block;
  y=ymax_block;
  while(x < xmax_block){
    xpos.push_back(x);
    ypos.push_back(y);
    x+=dx_tr;
  }

  // Right face
  x=xmax_block;
  y=ymax_block;
  while(y > ymin_block){
    xpos.push_back(x);
    ypos.push_back(y);
    y-=dx_tr;
  }

  // Bottom face
  x=xmax_block;
  y=ymin_block;
  while(x > xmin_block){
    xpos.push_back(x);
    ypos.push_back(y);
    x-=dx_tr;
  }

  // start at top left go clockwise across plane
  for(int index=0;index<xpos.size();index++){
    outp << xpos[index]  << " " << ypos[index]  << endl;
  }
  outp.close();
}

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

  double pi = 3.141592654;

  double L = .1833;  //cellSideLength = 1.0;
  double t = 0.00508; //cellWallThick  = 0.05;
  double H = 2.0;  //cellHeight     = 2.0;

  double rootX=0.,rootY=0.,rootZ=0.;

  /* Looking down from top, side 0 is bottom, side 1 is CCW, etc. */
  /* the "root" position is the center bottom of side 0           */

  vector<double> P1x(6), P2x(6), P3x(6), P4x(6);
  vector<double> P1y(6), P2y(6), P3y(6), P4y(6);
  vector<double> P1z(6), P2z(6), P3z(6), P4z(6);

  // Side 0
  P1x[0]=rootX-L/2;
  P1y[0]=rootY;
  P1z[0]=rootZ;

  P2x[0]=P1x[0];
  P2y[0]=P1y[0]+t;
  P2z[0]=P1z[0];

  P3x[0]=P1x[0];
  P3y[0]=P1y[0];
  P3z[0]=P1z[0]-H;

  P4x[0]=rootX+L/2;
  P4y[0]=rootY;
  P4z[0]=rootZ;

  // Side 1
  P1x[1]=P4x[0];
  P1y[1]=P4y[0];
  P1z[1]=P4z[0];

  P2x[1]=P1x[1]-t*cos(pi/6);
  P2y[1]=P1y[1]+t*sin(pi/6);
  P2z[1]=P1z[1];

  P3x[1]=P1x[1];
  P3y[1]=P1y[1];
  P3z[1]=P1z[1]-H;

  P4x[1]=P1x[1]+L*cos(pi/3);
  P4y[1]=P1y[1]+L*sin(pi/3);
  P4z[1]=P1z[1];

  // Side 2
  P1x[2]=P4x[1];
  P1y[2]=P4y[1];
  P1z[2]=P4z[1];

  P2x[2]=P1x[2]-t*cos(pi/6);
  P2y[2]=P1y[2]-t*sin(pi/6);
  P2z[2]=P1z[2];

  P3x[2]=P1x[2];
  P3y[2]=P1y[2];
  P3z[2]=P1z[2]-H;

  P4x[2]=rootX+L/2;
  P4y[2]=rootY+2.0*L*sin(pi/3);
  P4z[2]=rootZ;

  // Side 3
  P1x[3]=P4x[2];
  P1y[3]=P4y[2];
  P1z[3]=P4z[2];

  P2x[3]=P1x[3];
  P2y[3]=P1y[3]-t;
  P2z[3]=P1z[3];

  P3x[3]=P1x[3];
  P3y[3]=P1y[3];
  P3z[3]=P1z[3]-H;

  P4x[3]=rootX-L/2;
  P4y[3]=rootY+2.0*L*sin(pi/3);
  P4z[3]=rootZ;

  // Side 4
  P1x[4]=P4x[3];
  P1y[4]=P4y[3];
  P1z[4]=P4z[3];

  P2x[4]=P1x[4]+t*cos(pi/6);
  P2y[4]=P1y[4]-t*sin(pi/6);
  P2z[4]=P1z[4];

  P3x[4]=P1x[4];
  P3y[4]=P1y[4];
  P3z[4]=P1z[4]-H;

  P4x[4]=rootX-L/2-L*cos(pi/3);
  P4y[4]=rootY+L*sin(pi/3);
  P4z[4]=rootZ;

  // Side 5
  P1x[5]=P4x[4];
  P1y[5]=P4y[4];
  P1z[5]=P4z[4];

  P2x[5]=P1x[5]+t*sin(pi/6);
  P2y[5]=P1y[5]+t*cos(pi/6);
  P2z[5]=P1z[5];

  P3x[5]=P1x[5];
  P3y[5]=P1y[5];
  P3z[5]=P1z[5]-H;

  P4x[5]=rootX-L/2;
  P4y[5]=rootY;
  P4z[5]=rootZ;

  cout << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
  cout << "<Uintah_Include>" << endl;
  cout << " <union>" << endl;
  for(int i=0;i<3;i++){
   // the 0.7 coefficient below is a bit of a mystery to me
   double xoff = ((double) i)*(L+L*cos(pi/3)-0.7*t /*/sin(pi/6)*/);
   for(int j=0;j<3;j++){
    double yoff;
    if(i%2==0){
      yoff = ((double) j)*(2.*L*sin(pi/3)-t);
    } else {
      yoff = (L*sin(pi/3)-t/2.) + ((double) j)*(2.*L*sin(pi/3)-t);
    }
    for(int n=0;n<6;n++){
      cout << "  <parallelepiped label = \""<< i << "." << j << "side" << n << "\">" << endl;
      cout << "    <p1>[" << P1x[n]+xoff << "," << P1y[n]+yoff << "," << P1z[n] << "]</p1>" << endl;
      cout << "    <p2>[" << P2x[n]+xoff << "," << P2y[n]+yoff << "," << P2z[n] << "]</p2>" << endl;
      cout << "    <p3>[" << P3x[n]+xoff << "," << P3y[n]+yoff << "," << P3z[n] << "]</p3>" << endl;
      cout << "    <p4>[" << P4x[n]+xoff << "," << P4y[n]+yoff << "," << P4z[n] << "]</p4>" << endl;
      cout << "  </parallelepiped>" << endl;
    } // for n
   } // for j
  } // for i
  cout << " </union>" << endl;
  cout << "</Uintah_Include>" << endl;
}

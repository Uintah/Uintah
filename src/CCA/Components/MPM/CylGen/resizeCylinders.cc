// This code takes the output from cylgen (Position_Radius.txt) and attempts
// to increase the packing fraction by increasing each cylinder slightly,
// rechecking for intersections, and increasing further, or stopping if any
// overlap is detected.

// To compile:

// >g++ -O3 -o resizeCylinders resizeCylinders.cc

// To run:

// >resizeCylinder

// Note that this can be run successively (with diminishing returns) by
// moving the output from this program (Position_Radius.RS.txt) to the
// filename of the original distribution (Position_Radius.txt) and rerunning.

// i.e.

//  mv Position_Radius.RS.txt Position_Radius.txt

// >resizeCylinder

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

bool isCylCenterInsideRVE(double RVEsize,
                          double xCent, double yCent);

bool doesCylIntersectOthers(double partDia, vector<double> diaLocs,
                            double &gap,
                            double &xCent, double &yCent,
                            vector<double> xLocs,
                            vector<double> yLocs, int &i_this);

void printCylLocs(vector<double> xLocs,
                  vector<double> yLocs,
                  vector<double> diaLocs);

int main()
{

  // Parameters for user to change - BEGIN
  double RVEsize = 0.1;
  double diam_max = 0.011250;

  // Parameters for user to change - END IF AN EQUAL SIZE DISTRIBUTION IS O.K.

  // Part of optimizing the search for intersections
  int n_bins = RVEsize/diam_max;
  cout << "n_bins = " << n_bins << endl;

  //Store the locations in n_bins separate vectors, so we have a smaller region
  //to search for intersections
  vector<double> xLocs;
  vector<double> yLocs;
  vector<double> diaLocs;

  //Open file to receive cylinder descriptions
  string infile_name = "Position_Radius.txt";
  ifstream source(infile_name.c_str());
  if(!source){
    cerr << "File " << infile_name << " can't be opened." << endl;
  }
  double x,y,r;

  // Read in original cylinder description
  int outOfRVE=0;
  while(source >> x >> y >> r){
    if(isCylCenterInsideRVE(RVEsize,x,y)){
     xLocs.push_back(x);
     yLocs.push_back(y);
     diam_max=max(diam_max,2.*r);
     diaLocs.push_back(2.*r);
    } else{
      outOfRVE++;
    }
  }

  double total_cyl_area_orig = 0.0;
  double total_cyl_area_new  = 0.0;
  cout << xLocs.size() << endl;

  // Loop over all cylinders
  int numInts=0;
  for(int i = 0;i<xLocs.size();i++){
    double d = diaLocs[i];
    total_cyl_area_orig+= 0.25*M_PI*(d*d);;
    int i_this = i;
    double gap=9.e99;

    bool cylsIntersect = doesCylIntersectOthers(d, diaLocs, gap,
                                                xLocs[i],yLocs[i],
                                                xLocs,yLocs,i_this);

    if(cylsIntersect){
      numInts++;
    }
    if(gap>9.e90){
      gap = 0.0;
    }
    diaLocs[i] += 0.95*gap;
    d = diaLocs[i];
    diam_max=max(diam_max,d);
    total_cyl_area_new+= 0.25*M_PI*(d*d);;
  }

  cout << "numInts = " << numInts << endl;
  cout << "Cylinders out of RVE = " << outOfRVE << endl;
  cout << "Total cylinder area orig = " << total_cyl_area_orig << endl;
  cout << "Total cylinder area new  = " << total_cyl_area_new  << endl;
  cout << "New Maximum Diameter = " << diam_max << endl;

  printCylLocs(xLocs, yLocs, diaLocs);

}

bool isCylCenterInsideRVE(double RVEsize, double xCent, double yCent)
{

    // Find if the particle center is in the box
    if (xCent >= 0.0 && xCent <= RVEsize &&
        yCent >= 0.0 && yCent <= RVEsize) {
      return true;
    }
    return false;
}

bool doesCylIntersectOthers(double partDia, vector<double> diaLocs,
                            double &gap,
                            double &xCent, double &yCent,
                            vector<double> xLocs,
                            vector<double> yLocs, int &i_this)
{
  for(unsigned int i = 0; i<xLocs.size(); i++){
   if(i!=i_this){
    // Compute distance between centers
    double distCent = sqrt((xCent-xLocs[i])*(xCent-xLocs[i]) +
                           (yCent-yLocs[i])*(yCent-yLocs[i]));

    double sumRad = 0.5*(partDia + diaLocs[i]);

    double space = distCent - sumRad;
    gap = min(gap, space);

    if(space < 0.0){
      return true;
    }
   }
  }

  // None of the cyls intersected
  return false;
}

void printCylLocs(vector<double> xLocs, vector<double> yLocs,
                  vector<double> diaLocs)
{
  //Open file to receive cyl descriptions
  string outfile_name = "Test2D.RS.xml";
  ofstream dest(outfile_name.c_str());
  if(!dest){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  dest << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
  dest << "<Uintah_Include>" << endl;
  dest << "<union>\n\n";

  int cylcount = 0;
  for(unsigned int i = 0; i<xLocs.size(); i++){
    dest << "    <cylinder label = \"" << cylcount++ << "\">\n";
    dest << "       <top>[" << xLocs[i] << ", " << yLocs[i] << ", " << 10000 << "]</top>\n";
    dest << "       <bottom>[" << xLocs[i] << ", " << yLocs[i] << ", " << -10000.0 << "]</bottom>\n";
    dest << "       <radius>" << 0.5*diaLocs[i] << "</radius>\n";
    dest << "    </cylinder>\n";
  }

  dest << "</union>\n\n";
  dest << "</Uintah_Include>" << endl;

  string outfile_name2 = "Position_Radius.RS.txt";
  ofstream dest2(outfile_name2.c_str());
  if(!dest2){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  dest2.precision(15);

  for(unsigned int i = 0; i<xLocs.size(); i++){
    dest2 <<  xLocs[i] << " " << yLocs[i] << " " << 0.5*diaLocs[i] << "\n";
  }
}

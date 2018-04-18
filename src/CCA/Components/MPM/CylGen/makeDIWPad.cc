// This is a circle packing code that generates a distribution of cylinders
// according to a user specified size distribution and a target volume fraction.
// User can either ask for a uniform distribution of particles based on a
// min and max diameter, or can specify a non-uniform distribution.


// To compile:

//  g++ -O3 -o makeDIWPad makeDIWPad.cc

// To run:

// >makeDIWPad

// The code will create a PositionRadius.txt file, and also an xml file
// called DIWPad.xml which is compatible for inclusion in a Uintah Problem
// Specification (ups) file.


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

#if 0
void printCylLocs(vector<double>  xBot, vector<double> xTop,
                  vector<double>  yBot, vector<double> yTop,
                  vector<double>  zBot, vector<double> zTop,
                  double radius);
#endif

int main()
{

  ////////////////////////////////////////////////////////////////
  // Parameters for user to change - BEGIN

  int numLayers         = 10;
  int threadsPerLayer   = 20;
  double threadDiameter = 0.0355;
  double threadSpacing  = 0.0710;
  double threadOverlap  = 0.10; // Fraction of the diameter
  double threadLength   = threadSpacing*((double) (threadsPerLayer+0.0));
  string Qtr_or_Full = "Qtr";

  // No need to make changes below here
  ////////////////////////////////////////////////////////////////

  // Each thread needs a top, bottom and radius

  //Store the threads in vectors
  vector<double>  xTop,xBot;
  vector<double>  yTop,yBot;
  vector<double>  zTop,zBot;

  // Even layers run along the y-axis, odd layers run along the x-axis

  // Create threads for the even layers
  double z = 0.5*threadDiameter;
  for(int i=0;i<numLayers;i+=2){
    double x = 0.5*threadSpacing;
    for(int j=0;j<threadsPerLayer;j++){
      yBot.push_back(0.0);
      yTop.push_back(threadLength);
      xBot.push_back(x);
      xTop.push_back(x);
      zBot.push_back(z);
      zTop.push_back(z);
      x+=threadSpacing;
    } // Loop over threads in the layer
    z+=2.0*(threadDiameter)*(1.-threadOverlap);
  }   // Loop over the even layers

  // Create threads for the odd layers
//  double z = threadDiameter+0.5*threadDiameter-threadOverlap*threadDiameter;
  z = (1.5-threadOverlap)*threadDiameter;
  for(int i=1;i<numLayers;i+=2){
    double y = 0.5*threadSpacing;
    for(int j=0;j<threadsPerLayer;j++){
      xBot.push_back(0.0);
      xTop.push_back(threadLength);
      yBot.push_back(y);
      yTop.push_back(y);
      zBot.push_back(z);
      zTop.push_back(z);
      y+=threadSpacing;
    } // Loop over threads in the layer
    z+=2.0*(threadDiameter)*(1.-threadOverlap);
  }   // Loop over the even layers

#if 0
  printCylLocs(xBot, xTop, yBot, yTop, zBot, zTop, 0.5*threadDiameter);

}

void printCylLocs(vector<double>  xBot, vector<double> xTop,
                  vector<double>  yBot, vector<double> yTop,
                  vector<double>  zBot, vector<double> zTop, double radius)
{
#endif
  //Open file to receive cyl descriptions
  stringstream numL;
  numL << numLayers;
  string numLs = numL.str();
  stringstream TPL;
  TPL << threadsPerLayer;
  string TPLs = TPL.str();
  string outfile_name = "DIWPad.NL"+numLs+".TPL"+TPLs+".xml";
  ofstream dest(outfile_name.c_str());
  if(!dest){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  double topOfPad = zTop[zTop.size()-1] + 0.5*threadDiameter;

  dest << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
  dest << "<!--" << endl;
  dest << "  threadDiameter = " << threadDiameter << endl;
  dest << "  threadSpacing  = " << threadSpacing  << endl;
  dest << "  threadOverlap  = " << threadOverlap  << endl;
  dest << "  threadLength   = " << threadLength   << endl;
  dest << "  Qtr_or_Full    = " << Qtr_or_Full    << endl;
  dest << "  TOP OF PAD     = " << topOfPad       << endl;
  dest << "-->"  << endl;

  dest << "<Uintah_Include>" << endl;
  dest << "<union>\n";

  int cylcount = 0;
  for(unsigned int i = 0; i<xTop.size(); i++){
     dest << "  <cylinder label = \"" << cylcount++ << "\">\n";
     dest << "     <top>   [" << xBot[i] << ", " << yBot[i] << ", " << zBot[i] << "]</top>\n";
     dest << "     <bottom>[" << xTop[i] << ", " << yTop[i] << ", " << zTop[i] << "]</bottom>\n";
     dest << "     <radius>" << 0.5*threadDiameter << "</radius>\n";
     dest << "  </cylinder>\n";
  }

  dest << "</union>\n";
  dest << "</Uintah_Include>" << endl;

#if 0
  string outfile_name2 = "Position_Radius.txt";
  ofstream dest2(outfile_name2.c_str());
  if(!dest2){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  dest2.precision(15);

  for(int k=0;k<n_bins;k++){
   for(unsigned int i = 0; i<xLocs[k].size(); i++){
       dest2 <<  xLocs[k][i] << " " << yLocs[k][i] << " " << 0.5*diaLocs[k][i] << "\n";
   }
  }
#endif
}

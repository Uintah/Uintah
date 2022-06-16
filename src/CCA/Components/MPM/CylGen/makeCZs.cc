// Distribute cylinders into groups that are separated by a user specified
// distance.  A set of cylinders is provided in a file called
// Position_Radius.txt.  These are then printed into files in both xml format
// and in plain text
// Read in LineSegments files for different groups and look for segment pairs 
// (from different groups) that are within a specified distance of each other

// To compile:

//                  g++ -O3 -o makeCZs makeCZs.cc

// To run:

//                  makeCZs


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;

int main()
{
  // Parameters for user to change - BEGIN
  int numGroups = 5;  // 0 - 4
  int numfiles[5] = {33,35,37,29,9};

  vector<double> xLS[numGroups];
  vector<double> yLS[numGroups];
  vector<double> LSvx[numGroups];
  vector<double> LSvy[numGroups];

  // Parameters for user to change - END

  //Open file to receive sphere descriptions
  string outfile_name = "CohesiveZones.txt";
  ofstream dest(outfile_name.c_str());
  if(!dest){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  // Read in all of the line segments store by group in STL vectors
  for(int i = 0; i < numGroups; i++){
    for(int j = 0; j <= numfiles[i]; j++){
      stringstream group, file;
      string  g,f;
      group << i;
      file << j;
      g = group.str();
      f = file.str();
      //Open file to receive sphere descriptions
      string infile_name = "LS." + f + "." + g + ".txt";
      ifstream source(infile_name.c_str());
      if(!source){
        cerr << "File " << infile_name << " can't be opened." << endl;
      }
      std::string junk;
      std::getline(source, junk); // skip the first line
      double time, px, py, pz, lsvx, lsvy, lsvz;
      int pI, mat;
      while(source >> time >> pI >> mat >> px >> py >> pz >> lsvx >> lsvy >> lsvz){
        xLS[i].push_back(px);
        yLS[i].push_back(py);
        LSvx[i].push_back(lsvx);
        LSvy[i].push_back(lsvy);
      }
    }
    cout << "xLS[" << i << "].size() = " << xLS[i].size() << endl;
  }

  vector<double> CZx;
  vector<double> CZy;
  vector<double> CZtm;
  vector<double> CZbm;

  for(int i = 0; i < numGroups; i++){
    cout << "working on group " << i << endl;
    for(int j = i+1; j < numGroups; j++){
      for(int k = 0; k < xLS[i].size(); k++){
        for(int l = 0; l < xLS[j].size(); l++){
          double sep = sqrt((xLS[i][k] - xLS[j][l])*(xLS[i][k] - xLS[j][l]) +
                            (yLS[i][k] - yLS[j][l])*(yLS[i][k] - yLS[j][l]));
          if(sep < 1.e-4){
            CZx.push_back(0.5*(xLS[i][k]+xLS[j][l]));
            CZy.push_back(0.5*(yLS[i][k]+yLS[j][l]));
            CZtm.push_back(i);
            CZbm.push_back(j);
          }
        }
      }
    }
  }

  for(int i = 0; i < CZx.size(); i++){
    dest << CZtm[i] << " " << CZbm[i] << " " << CZx[i] << " " << CZy[i] << " " << endl;
  }

#if 0
  vector<double> xLocs;
  vector<double> yLocs;
  vector<double> dLocs;
  vector<int>    grouped;

  // Read in and sort the cylinders according the sizes specified above.
  double x,y,r,d;
  while(source >> x >> y >> r){
   d=2.*r;
   xLocs.push_back(x);
   yLocs.push_back(y);
   dLocs.push_back(d);
   grouped.push_back(0);
  }

  vector<vector<double> > xgroupLocs(maxGroups);
  vector<vector<double> > ygroupLocs(maxGroups);
  vector<vector<double> > dgroupLocs(maxGroups);
  
  int igroup = 0;
  int numGrouped = 1;
  while(numGrouped<xLocs.size()){
  for(int j = 0; j<xLocs.size(); j++){
   if(grouped[j]==0){
    if(xgroupLocs[igroup].size()==0){
      xgroupLocs[igroup].push_back(xLocs[j]);
      ygroupLocs[igroup].push_back(yLocs[j]);
      dgroupLocs[igroup].push_back(dLocs[j]);
      grouped[j]=1;
      numGrouped++;
    }
    double space = RVEsize;
    for(int k = 0; k<xgroupLocs[igroup].size(); k++){
      double xCent = xgroupLocs[igroup][k];
      double yCent = ygroupLocs[igroup][k];
      double partDia = dgroupLocs[igroup][k];
      double distCent = sqrt((xCent-xLocs[j])*(xCent-xLocs[j]) +
                             (yCent-yLocs[j])*(yCent-yLocs[j]));
      double sumRad = 0.5*(partDia + dLocs[j]);
      space = min(space, distCent - sumRad);
    }
    if(space > minSep) {
      xgroupLocs[igroup].push_back(xLocs[j]);
      ygroupLocs[igroup].push_back(yLocs[j]);
      dgroupLocs[igroup].push_back(dLocs[j]);
      grouped[j]=1;
      numGrouped++;
    }
   } // not grouped yet
  }
  cout << "numGrouped = " << numGrouped << endl;
  igroup++;
  }
  cout << "igroup = " << igroup << endl;

#endif
#if 0
  // Bin the cylinders according to their x-position.  No real good reason
  // to do this other than compatibility with an existing print function
  for(int i = 0; i<igroup; i++){
    printCylLocs(xgroupLocs[i],ygroupLocs[i],dgroupLocs[i],RVEsize,i);
  }
#endif

}

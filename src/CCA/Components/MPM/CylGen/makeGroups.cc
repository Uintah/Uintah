// Distribute cylinders into groups that are separated by a user specified
// distance.  A set of cylinders is provided in a file called
// Position_Radius.txt.  These are then printed into files in both xml format
// and in plain text

// To compile:

//                  g++ -O3 -o makeGroups makeGroups.cc

// To run:

//                  makeGroups


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

void printCylLocs(vector<double> xLocs, vector<double> yLocs,
                  vector<double> diaLocs, double RVEsize, int groupNum);

bool isCylinderTooCloseToOthers(double partDia, vector<double> diaLocs,
                                double &minSep,
                                double &xCent, double &yCent,
                                vector<double> xLocs,
                                vector<double> yLocs, int &i_this);

int main()
{
  // Parameters for user to change - BEGIN
  double RVEsize = 0.1;
  double minSep = 0.04*RVEsize;
  int maxGroups = 20;

  // Parameters for user to change - END

  //Open file to receive sphere descriptions
  string infile_name = "Position_Radius.txt";
  ifstream source(infile_name.c_str());
  if(!source){
    cerr << "File " << infile_name << " can't be opened." << endl;
  }

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

#if 1
  // Bin the cylinders according to their x-position.  No real good reason
  // to do this other than compatibility with an existing print function
  for(int i = 0; i<igroup; i++){
    printCylLocs(xgroupLocs[i],ygroupLocs[i],dgroupLocs[i],RVEsize,i);
  }
#endif

}

void printCylLocs(vector<double> xLocs,
                  vector<double> yLocs,
                  vector<double> diaLocs,
                  double RVEsize, int groupNum)
{
    stringstream out;
    string  s;

    out << groupNum;
    s = out.str();
   
    //Open file to receive sphere descriptions
    string outfile_name = "Test2D." + s + ".xml";
    ofstream dest(outfile_name.c_str());
    if(!dest){
      cerr << "File " << outfile_name << " can't be opened." << endl;
    }

    dest << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
    dest << "<Uintah_Include>" << endl;
    dest << "<intersection>\n";
    dest << "  <box>\n";
    dest << "    <min>[0.0, 0.0, -10000.0]</min>" << endl;
    dest << "    <max>[" << RVEsize << ", " << RVEsize << ",  10000.0]</max>" << endl;
    dest << "  </box>\n\n";

    dest << "  <union>\n";

    if(xLocs.size()>0){
      for(unsigned int i = 0; i<xLocs.size(); i++){
         dest << "    <cylinder>\n";
         dest << "       <top>[" << xLocs[i] << ", " << yLocs[i] << ", " << 10000 << "]</top>\n";
         dest << "       <bottom>[" << xLocs[i] << ", " << yLocs[i] << ", " << -10000.0 << "]</bottom>\n";
         dest << "       <radius>" << 0.5*diaLocs[i] << "</radius>\n";
         dest << "    </cylinder>\n";
      }
    }
    dest << "  </union>\n\n";
    dest << " </intersection>\n\n";

    dest << "</Uintah_Include>" << endl;

    string outfile_name2 = "Position_Radius." + s + ".txt";
    ofstream dest2(outfile_name2.c_str());
    if(!dest2){
      cerr << "File " << outfile_name << " can't be opened." << endl;
    }

    if(xLocs.size()>0){
      for(unsigned int i = 0; i<xLocs.size(); i++){
        dest2 <<  xLocs[i] << " " << yLocs[i] << " " << 0.5*diaLocs[i] << "\n";
      }
    }
}

bool isCylinderTooCloseToOthers(double partDia, vector<double> diaLocs,
                                double &minSep, double &xCent, double &yCent,
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

    if(space < 0.0){
      return true;
    }
   }
  }

  // None of the cyls intersected
  return false;
}


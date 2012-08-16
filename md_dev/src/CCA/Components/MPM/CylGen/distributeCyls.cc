// Distribute cylinders into various materials according to a user specified
// size distribution.  A set of cylinders is provided in a file called
// Position_Radius.txt.  These are then printed into files in both xml format
// and in plain text

// To compile:

//                  g++ -O3 -o distributeCyls distributeCyls.cc

// To run:

//                  distributeCyls


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

void printCylLocs(vector<vector<double> > xLocs, vector<vector<double> > yLocs,
                  vector<vector<double> > diaLocs,
                  int n_bins, double RVEsize, double diam_max, string matl_name,
                  int numFields);

int main()
{
  // Parameters for user to change - BEGIN
  double RVEsize = 0.1;
  double diam_max = 0.01375;
  int n_bins  = 10;
  int n_sizes = 4;
  int n_matls = 3;
  double sizes[n_sizes];
  int num_each_size[n_sizes];

  // The particles will be binned according to size and material.
  // The following prescribes the sizes according to which cylinders
  // will be sorted.  That is, cylinders larger than size[i]
  // will go into bin i.
  sizes[0]=0.0100; num_each_size[0]=0;
  sizes[1]=0.0075; num_each_size[1]=0;
  sizes[2]=0.0050; num_each_size[2]=0;
  sizes[3]=0.0006; num_each_size[3]=0;

  string matl_name[3];
  matl_name[0] = "W";
  matl_name[1] = "Cu";
  matl_name[2] = "Pb";

  // matl_size is the cumulative volume fraction of cylinders for a given
  // material at the sizes specified in the sizes array above
  double matl_size[3][n_sizes];
  matl_size[0][0]=0.0;
  matl_size[1][0]=0.08;
  matl_size[2][0]=1.0;

  matl_size[0][1]=0.54;
  matl_size[1][1]=0.65;
  matl_size[2][1]=1.0;

  matl_size[0][2]=0.81;
  matl_size[1][2]=0.89;
  matl_size[2][2]=1.0;

  matl_size[0][3]=0.64;
  matl_size[1][3]=0.87;
  matl_size[2][3]=1.0;

  // For MPM, it is desirable to prevent particles from the same "field" from
  // touching.  For this reason, even though there are three matls, their
  // grains are spread among, in this case, 10 fields
  int num_fields[n_matls];
  num_fields[0]=6;
  num_fields[1]=2;
  num_fields[2]=2;

  // Parameters for user to change - END

  //Open file to receive sphere descriptions
  string infile_name = "Position_Radius.txt";
  ifstream source(infile_name.c_str());
  if(!source){
    cerr << "File " << infile_name << " can't be opened." << endl;
  }

  double x,y,r,d;

  vector<vector<double> > xsizeLocs(n_sizes);
  vector<vector<double> > ysizeLocs(n_sizes);
  vector<vector<double> > dsizeLocs(n_sizes);

  // Read in and sort the cylinders according the sizes specified above.
  while(source >> x >> y >> r){
   d=2.*r;
   if(d>sizes[n_sizes-1]){
     int i=0;
     while(sizes[i]>d){
       i++;
     }
     num_each_size[i]++;
     xsizeLocs[i].push_back(x);
     ysizeLocs[i].push_back(y);
     dsizeLocs[i].push_back(d);
   }
  }

  cout << "Number of each size, exclusively" << endl;
  for(int i=0;i< n_sizes;i++){
    cout << "There are " << num_each_size[i] << " cylinders larger than " << sizes[i] << endl;
  }

  vector<vector<double> > xmatlLocs(n_matls);
  vector<vector<double> > ymatlLocs(n_matls);
  vector<vector<double> > dmatlLocs(n_matls);
  
  // Distribute the sorted cylinders to the appropriate materials based on
  // a random selection method
  // For each size, loop over the cylinders in that size's bin
  // A random # (RN) is generated, that will be between 0 and 1.  For that
  // RN, determine which matl's vector will get that particular cylinder,
  // then move on to the next cylinder.
  for(int n = 0; n<n_sizes; n++){
    for(int k = 0; k<xsizeLocs[n].size(); k++){
      double RN = drand48();
      int i=0;
      while(RN>matl_size[i][n]){
        i++;
      }
      xmatlLocs[i].push_back(xsizeLocs[n][k]);
      ymatlLocs[i].push_back(ysizeLocs[n][k]);
      dmatlLocs[i].push_back(dsizeLocs[n][k]);
    }
  }

  // Just some informative diagnostics
  for(int i = 0; i<n_matls; i++){
    cout << "i = " << i << endl;
    cout << dmatlLocs[i].size() << endl;
  }

  // Bin the cylinders according to their x-position.  No real good reason
  // to do this other than compatibility with an existing print function
  for(int i = 0; i<n_matls; i++){
    vector<vector<double> > xbinLocs(n_bins);
    vector<vector<double> > ybinLocs(n_bins);
    vector<vector<double> > dbinLocs(n_bins);

    for(int k = 0; k<xmatlLocs[i].size(); k++){
      int index = (xmatlLocs[i][k]/RVEsize)*((double) n_bins);
      xbinLocs[index].push_back(xmatlLocs[i][k]);
      ybinLocs[index].push_back(ymatlLocs[i][k]);
      dbinLocs[index].push_back(dmatlLocs[i][k]);
    }

    printCylLocs(xbinLocs,ybinLocs,dbinLocs,n_bins,RVEsize,diam_max,matl_name[i],num_fields[i]);
  }

}

void printCylLocs(vector<vector<double> > xLocs,
                 vector<vector<double> > yLocs,
                 vector<vector<double> > diaLocs, int n_bins,
                 double RVEsize, double diam_max, string matl_name,
                 int numFields)
{
  int spherecount = 0;
  for(int p=0;p<numFields;p++){

    stringstream out;
    string  s;

    out << p;
    s = out.str();
   
    //Open file to receive sphere descriptions
    string outfile_name = "Test2D." + matl_name + s + ".xml";
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
    for(int k=0;k<n_bins;k++){
      if(xLocs[k].size()>0){
        for(unsigned int i = 0; i<xLocs[k].size(); i++){
          if(i%numFields==p){
             dest << "    <cylinder>\n";
             dest << "       <top>[" << xLocs[k][i] << ", " << yLocs[k][i] << ", " << 10000 << "]</top>\n";
             dest << "       <bottom>[" << xLocs[k][i] << ", " << yLocs[k][i] << ", " << -10000.0 << "]</bottom>\n";
             dest << "       <radius>" << 0.5*diaLocs[k][i] << "</radius>\n";
             dest << "    </cylinder>\n";
          }
        }
      }
    }
    dest << "  </union>\n\n";
    dest << " </intersection>\n\n";

    dest << "</Uintah_Include>" << endl;

    string outfile_name2 = "Position_RadiusNew." + matl_name + s + ".txt";
    ofstream dest2(outfile_name2.c_str());
    if(!dest2){
      cerr << "File " << outfile_name << " can't be opened." << endl;
    }

    for(int k=0;k<n_bins;k++){
      if(xLocs[k].size()>0){
        for(unsigned int i = 0; i<xLocs[k].size(); i++){
          if(i%numFields==p){
         dest2 <<  xLocs[k][i] << " " << yLocs[k][i] << " " << 0.5*diaLocs[k][i] << "\n";
          }
        }
      }
    }
  }
}

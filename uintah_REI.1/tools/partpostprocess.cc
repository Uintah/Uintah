#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

//  This is a standalone program (g++ partpostprocess.cc) that
//  can read in the datafiles like those created by the puda -jim1
//  option.

//  The goal is to track a subset of variables for a particular particle
//  or set of particles, in this case particles that are initially along
//  the line z=.001, y=.183.  This is done by creating a vector of
//  particleIDs associated with those particles by scanning the partout0000
//  file, and then finding the  particles with those same IDs in subsequent
//  partout* files.  This data is then printed to files.

int main()
{
  double x,y,z,velx,vely,velz;
  int64_t pID;

  vector<int64_t> pIDs;
  vector<double>  xs,velxs,velys,velzs;

  string fileroot = "partout";
  int filenum = 0;
  char fnum[5];
  sprintf(fnum,"%04d",filenum);
  string file_name = fileroot+fnum;
  ifstream source(file_name.c_str());
  if(!source){
    cerr << "File " << file_name << " can't be opened." << endl;
  }
  
  while(source >> x >> y >> z >> velx >> vely >> velz >> pID){
     if(z ==.001 && y == 0.183){
        pIDs.push_back(pID);
     }
  }

  for(int i=0;i<pIDs.size();i++){
    cout << pIDs[i] << endl;
  }

  int step_max=100;
  for(int step=1;step<step_max;step++){
    string fileroot    = "partout";
    string outfileroot = "pvels";
    char fnum[5];
    sprintf(fnum,"%04d",step);
    string file_name    = fileroot+fnum;
    string outfile_name = outfileroot+fnum;
    ifstream source(file_name.c_str());
    if(!source){
      cerr << "File " << file_name << " can't be opened." << endl;
      exit(1);
    }
    ofstream dest(outfile_name.c_str());
    if(!dest){
      cerr << "File " << outfile_name << " can't be opened." << endl;
    }

    while(source >> x >> y >> z >> velx >> vely >> velz >> pID){
      for(int i=0;i<pIDs.size();i++){
        if(pID==pIDs[i]){
           xs.push_back(x);
           velxs.push_back(velx);
           velys.push_back(vely);
           velzs.push_back(velz);
           i+=100000; 
        }  // if
      }  // for
    }  // while
    for(int i=0;i<xs.size();i++){
      dest << xs[i] << " " << velxs[i] << " " << velys[i]
                    << " " << velzs[i] << endl;
    }  // for
    xs.clear();
    velxs.clear();
    velys.clear();
    velzs.clear();
  }
}

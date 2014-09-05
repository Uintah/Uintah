/*
 *  extractS.cc: Print out a uintah data archive for particle stress data
 *
 *  Written by:
 *   Biswajit Banerjee
 *   July 2004
 *
 */

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/Array3.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

typedef struct{
  vector<Matrix3> stress;
  vector<Point> px;
  vector<long64> id;
  vector<double> time;
  vector<int> patch;
  vector<int> matl;
} MaterialData;

void usage(const std::string& badarg, const std::string& progname);

void printStress(DataArchive* da, 
                 int matID,
                 vector<long64>& partID,
                 string outFile,
		 bool timeFiles);

int main(int argc, char** argv)
{
  string partVar;
  int matID = -1;
  string partIDFile;
  string udaDir;
  string outFile;
  bool timeFiles = false;

  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if (s == "-m") {
      string id = argv[++i];
      if (id[0] == '-') 
        matID = -1;
      else
        matID = atoi(argv[i]);
    } else if(s == "-p"){
      partIDFile = argv[++i];
      if (partIDFile[0] == '-') 
        usage("-p <particle id file>", argv[0]);
    } else if(s == "-uda"){
      udaDir = argv[++i];
      if (udaDir[0] == '-') 
        usage("-uda <archive file>", argv[0]);
    } else if(s == "-o"){
      outFile = argv[++i];
      if (outFile[0] == '-') 
        usage("-o <output file>", argv[0]);
    } else if (s == "-timefiles") {
      timeFiles = true;
    } 
  }
  if (argc < 9) usage( "", argv[0] );

  cout << "Particle Variable to be extracted = p.stress\n";
  cout << "Material ID to be extracted = " << matID << endl;

  // Read the particle ID file
  cout << "Particle ID File to be read = " << partIDFile << endl;
  vector<long64> partID;
  ifstream pidFile(partIDFile.c_str());
  if (!pidFile.is_open()) {
    cerr << "Particle ID File " << partIDFile << " not found \n";
    exit(1);
  }
  do {
    double t = 0.0, x = 0.0, y = 0.0, z = 0.0;
    int patch = 0, mat = 0;
    long64 id = 0;
    pidFile >> t >> patch >> mat >> id >> x >> y >> z;
    partID.push_back(id);
  } while (!pidFile.eof());
  
  cout << "  Number of Particle IDs = " << partID.size() << endl;
  for (unsigned int ii = 0; ii < partID.size()-1 ; ++ii) {
    cout << "    p"<< (ii+1) << " = " << partID[ii] << endl;
  }

  cout << "Output file name = " << outFile << endl;
  cout << "UDA directory to be read = " << udaDir << endl;
  try {
    DataArchive* da = scinew DataArchive(udaDir);
    
    // Print a particular particle variable
    printStress(da, matID, partID, outFile, timeFiles);
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}
void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "") cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname 
       << " -m <material id>"
       << " -p <particle id file>"
       << " -uda <archive file>"
       << " -o <output file>\n\n";
  exit(1);
}

////////////////////////////////////////////////////////////////////////////
//
// Print a particle variable
//
////////////////////////////////////////////////////////////////////////////
void printStress(DataArchive* da, 
                 int matID,
                 vector<long64>& partID,
                 string outFile,
		 bool timeFiles)
{

  // Check if the particle variable is available
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  string partVar("p.stress");
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == partVar) variableFound = true;
  }
  if (!variableFound) {
    cerr << "Variable " << partVar << " not found\n"; 
    exit(1);
  }

  // Create arrays of material data for each particle ID
  MaterialData* matData = scinew MaterialData[partID.size()-1];
  //for (unsigned int ii = 0; ii < partID.size() ; ++ii) {
  //  matData[ii] = scinew MaterialData();
  //}

  // Now that the variable has been found, get the data for all 
  // available time steps from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
      
  // Loop thru all the variables 
  for(int v=0;v<(int)vars.size();v++){
    std::string var = vars[v];

    // Find the name of the variable
    if (var == partVar) {

      // Loop thru all time steps 
      int startPatch = 1;
      for(unsigned long t=0;t<times.size();t++){
        double time = times[t];
        cerr << "t = " << time ;
        clock_t start = clock();
        GridP grid = da->queryGrid(time);

	unsigned int numFound = 0;

        // Loop thru all the levels
        for(int l=0;l<grid->numLevels();l++){
          if (numFound == partID.size()-1) break;

          LevelP level = grid->getLevel(l);
          Level::const_patchIterator iter = level->patchesBegin(); 
          int patchIndex = 0;

          // Loop thru all the patches
          for(; iter != level->patchesEnd(); iter++){
            if (numFound == partID.size()-1) break;

            const Patch* patch = *iter;
            ++patchIndex; 
            if (patchIndex < startPatch) continue;

            ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
            ConsecutiveRangeSet::iterator matlIter = matls.begin(); 

            // loop thru all the materials
            for(; matlIter != matls.end(); matlIter++){
              if (numFound == partID.size()-1) break;

              int matl = *matlIter;
              if (matID == -1 || matl == matID || matl == matID+1) {
		ParticleVariable<Matrix3> value;
		da->query(value, var, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		if(pset->numParticles() > 0){
		  ParticleVariable<long64> pid;
		  da->query(pid, "p.particleID", matl, patch, time);
		  ParticleVariable<Point> px;
		  da->query(px, "p.x", matl, patch, time);
		  vector<bool> found;
		  for (unsigned int ii = 0; ii < partID.size()-1 ; ++ii) {
		    found.push_back(false);
		  }
		  ParticleSubset::iterator iter = pset->begin();
		  for(;iter != pset->end(); iter++){
		    for (unsigned int ii = 0; ii < partID.size()-1 ; ++ii) {
		      if (found[ii]) continue;
		      if (partID[ii] != pid[*iter]) continue;
		      matData[ii].stress.push_back(value[*iter]);
		      matData[ii].id.push_back(pid[*iter]);
		      matData[ii].px.push_back(px[*iter]);
		      matData[ii].time.push_back(time);
		      matData[ii].patch.push_back(patchIndex);
		      matData[ii].matl.push_back(matl);
		      found[ii] = true;
		      ++numFound;
		      break;
		    }
		    if (numFound == partID.size()-1) break;
		  }
                  if (numFound > 0 && startPatch == 0) startPatch = patchIndex;
		}
              } // end of mat compare if
            } // end of material loop
          } // end of patch loop
        } // end of level loop
        clock_t end = clock();
        double timetaken = (double) (end - start)/(double) CLOCKS_PER_SEC;
        cerr << " CPU Time = " << timetaken << " s" << " found " 
             << numFound << endl;
      } // end of time step loop
    } // end of var compare if
  } // end of variable loop

  if (timeFiles) {
    // Create output files for each of the timesteps
    for(unsigned long jj=0;jj<times.size();jj++){
      double time = times[jj];
      ostringstream name;
      name << outFile << "_t" << setw(2) << setfill('0') << (jj+1);
      ofstream file(name.str().c_str());
      file.setf(ios::scientific,ios::floatfield);
      file.precision(8);
      cout << "Created output file " << name.str() << " for time "
           << time << endl;
      for (unsigned int ii = 0; ii < partID.size()-1 ; ++ii) {
        int patchIndex = matData[ii].patch[jj];
        int matl = matData[ii].matl[jj];
        long64 pid = matData[ii].id[jj];
        Matrix3 sig = matData[ii].stress[jj];
	Point px = matData[ii].px[jj];
        file << time << " " << patchIndex << " " << matl ;
        file << " " << pid;
	file << " " << px.x() << " " << px.y() << " " << px.z();
        file << " " << sig(0,0) << " " << sig(1,1) << " " << sig(2,2)
  	     << " " << sig(1,2) << " " << sig(2,0) << " " << sig(0,1) << endl;
      }
      file.close();
    }
  } else {
    // Create output files for each of the particle IDs
    for (unsigned int ii = 0; ii < partID.size()-1 ; ++ii) {
      ostringstream name;
      name << outFile << "_p" << setw(2) << setfill('0') << (ii+1);
      ofstream file(name.str().c_str());
      file.setf(ios::scientific,ios::floatfield);
      file.precision(8);
      cout << "Created output file " << name.str() << " for particle ID "
           << partID[ii] << endl;
      for (unsigned int jj = 0; jj < matData[ii].time.size(); ++jj) {
        double time = matData[ii].time[jj];
        int patchIndex = matData[ii].patch[jj];
        int matl = matData[ii].matl[jj];
        long64 pid = matData[ii].id[jj];
        Matrix3 sig = matData[ii].stress[jj];
	Point px = matData[ii].px[jj];
        file << time << " " << patchIndex << " " << matl ;
        file << " " << pid;
	file << " " << px.x() << " " << px.y() << " " << px.z();
        file << " " << sig(0,0) << " " << sig(1,1) << " " << sig(2,2)
  	     << " " << sig(1,2) << " " << sig(2,0) << " " << sig(0,1) << endl;
      }
      file.close();
    }
  }
}


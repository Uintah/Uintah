/*
 *  extractV.cc: Print out a uintah data archive for particle 
 *  deformation gradient data
 *
 *  Written by:
 *   Biswajit Banerjee
 *   July 2004
 *
 */

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <Core/Thread/Mutex.h>
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
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

Mutex cerrLock( "cerrLock" );

typedef struct{
  vector<Matrix3> defGrad;
  vector<long64> id;
  vector<double> time;
  vector<int> patch;
  vector<int> matl;
} MaterialData;

void usage(const std::string& badarg, const std::string& progname);

void printDefGrad(DataArchive* da, 
                  int matID,
                  vector<long64>& partID,
                  string outFile);

int main(int argc, char** argv)
{
  string partVar;
  int matID = 0;
  string partIDFile;
  string udaDir;
  string outFile;

  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if (s == "-materialID") {
      string id = argv[++i];
      if (id[0] == '-') 
        usage("-materialID <material id>", argv[0]);
      matID = atoi(argv[i]);
    } else if(s == "-partIDFile"){
      partIDFile = argv[++i];
      if (partIDFile[0] == '-') 
        usage("-partIDFile <particle id file>", argv[0]);
    } else if(s == "-udaDir"){
      udaDir = argv[++i];
      if (udaDir[0] == '-') 
        usage("-udaDir <archive file>", argv[0]);
    } else if(s == "-outputFile"){
      outFile = argv[++i];
      if (outFile[0] == '-') 
        usage("-outFile <output file>", argv[0]);
    } 
  }
  if (argc != 9) usage( "", argv[0] );

  cout << "Particle Variable to be extracted = p.deformationMeasure\n";
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
    long64 id = 0;
    pidFile >> id;
    partID.push_back(id);
  } while (!pidFile.eof());
  
  cout << "  Number of Particle IDs = " << partID.size() << endl;
  for (unsigned int ii = 0; ii < partID.size() ; ++ii) {
    cout << "    p"<< (ii+1) << " = " << partID[ii] << endl;
  }

  cout << "Output file name = " << outFile << endl;
  cout << "UDA directory to be read = " << udaDir << endl;
  try {
    DataArchive* da = scinew DataArchive(udaDir);
    
    // Print a particular particle variable
    printDefGrad(da, matID, partID, outFile);
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
       << " -materialID <material id>"
       << " -partidFile <particle id file>"
       << " -udaDir <archive file>"
       << " -outputFile <output file>\n";
  cerr << " -- Reads only Vector and Matrix3 data -- \n";
  exit(1);
}

////////////////////////////////////////////////////////////////////////////
//
// Print a particle variable
//
////////////////////////////////////////////////////////////////////////////
void printDefGrad(DataArchive* da, 
                  int matID,
                  vector<long64>& partID,
                  string outFile){

  // Check if the particle variable is available
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  string partVar("p.deformationMeasure");
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == partVar) variableFound = true;
  }
  if (!variableFound) {
    cerr << "Variable " << partVar << " not found\n"; 
    exit(1);
  }

  // Create arrays of material data for each particle ID
  MaterialData *matData = scinew MaterialData[partID.size()];
  //for (unsigned int ii = 0; ii < partID.size() ; ++ii) {
  //  matData[ii] = scinew MaterialData;
  //}

  // Now that the variable has been found, get the data for all 
  // available time steps from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
      
  // Loop thru all time steps 
  for(unsigned long t=0;t<=times.size();t++){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(time);

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patchIterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        ++patchIndex; 

        // Loop thru all the variables 
        for(int v=0;v<(int)vars.size();v++){
          std::string var = vars[v];
          const Uintah::TypeDescription* td = types[v];
          const Uintah::TypeDescription* subtype = td->getSubType();

          // Check if the variable is a ParticleVariable
          if(td->getType() == Uintah::TypeDescription::ParticleVariable) { 

            // loop thru all the materials
            ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
            ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
            for(; matlIter != matls.end(); matlIter++){
              int matl = *matlIter;

              if (matl == matID || matl == matID+1) {

                // Find the name of the variable
                if (var == partVar) {
                  //cout << "Material: " << matl << endl;
                  switch(subtype->getType()){
                  case Uintah::TypeDescription::Matrix3:
                    {
                      ParticleVariable<Matrix3> value;
                      da->query(value, var, matl, patch, time);
                      ParticleSubset* pset = value.getParticleSubset();
                      if(pset->numParticles() > 0){
                        ParticleVariable<long64> pid;
                        da->query(pid, "p.particleID", matl, patch, time);
                        ParticleSubset::iterator iter = pset->begin();
                        for(;iter != pset->end(); iter++){
                          for (unsigned int ii = 0; ii < partID.size() ; ++ii) {
                            if (partID[ii] != pid[*iter]) continue;
			    matData[ii].defGrad.push_back(value[*iter]);
			    matData[ii].id.push_back(pid[*iter]);
			    matData[ii].time.push_back(time);
			    matData[ii].patch.push_back(patchIndex);
			    matData[ii].matl.push_back(matl);
                          }
                        }
                      }
                    }
                  break;
                  default:
                    break;
                  }
                } // end of var compare if
              } // end of mat compare if
            } // end of material loop
          } // end of ParticleVariable if
        } // end of variable loop
      } // end of patch loop
    } // end of level loop
  } // end of time step loop

  // Create output files for each of the particle IDs
  for (unsigned int ii = 0; ii < partID.size() ; ++ii) {
    ostringstream name;
    name << outFile << "_p" << setw(2) << setfill('0') << ii;
    ofstream file(name.str().c_str());
    file.setf(ios::scientific,ios::floatfield);
    file.precision(8);
    cout << "Created output file " << name << " for particle ID "
         << partID[ii] << endl;
    for (unsigned int jj = 0; jj < matData[ii].time.size(); ++jj) {
      double time = matData[ii].time[jj];
      int patchIndex = matData[ii].patch[jj];
      int matl = matData[ii].matl[jj];
      long64 pid = matData[ii].id[jj];
      Matrix3 defGrad = matData[ii].defGrad[jj];
      file << time << " " << patchIndex << " " << matl ;
      file << " " << pid;
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          file << " " << defGrad(kk,ll) ;
        }
      }
      file << endl;
    }
    file.close();
  }
}


/*
 *  selectpart.cc: Get the range of particle ids in a given selection box
 *
 *  Written by:
 *   Biswajit Banerjee
 *   July 2005
 *
 */

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/SymmMatrix3.h>
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

// declarations
void usage(const std::string& badarg, const std::string& progname);
void printParticleID(DataArchive* da, const Box& box);

// Main
int main(int argc, char** argv)
{
  /*
   * Default values
   */
  double xmin = 0.0, ymin = 0.0, zmin = 0.0;
  double xmax = 1.0, ymax = 1.0, zmax = 1.0;
  string filebase;

  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-box"){
      xmin = atof(argv[++i]);
      ymin = atof(argv[++i]);
      zmin = atof(argv[++i]);
      xmax = atof(argv[++i]);
      ymax = atof(argv[++i]);
      zmax = atof(argv[++i]);
    } 
  }
  filebase = argv[argc-1];

  if(filebase == "" || filebase == argv[0]){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    DataArchive* da = scinew DataArchive(filebase);

    // Setup box
    Point lower(xmin,ymin,zmin);
    Point upper(xmax,ymax,zmax);
    Box box(lower, upper);
    
    // Get the particle IDs
    printParticleID(da, box);

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

  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -box <xmin> <ymin> <zmin> <xmax> <ymax> <zmax>\n";
  exit(1);
}

////////////////////////////////////////////////////////////////////////////
//
// Print particle IDs
//
////////////////////////////////////////////////////////////////////////////
void printParticleID(DataArchive* da, const Box& box)
{
  // Box
  cerr << "Selction box = " << box << endl;

  // Check if the particle variable p.particleId and p.x are available
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == "p.particleID" || var == "p.x") variableFound = true;
  }
  if (!variableFound) {
    cerr << "p.particleID or p.x not found\n"; 
    exit(1);
  }

  // Now that the variable has been found, get the data for the
  // first timestep from the archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
      
  // Get the data for the first timestep
  double time = times[0];
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

      // Search for p.x
      std::string var = "p.x";

      // loop thru all the materials
      ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
      ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
      for(; matlIter != matls.end(); matlIter++){
        int matl = *matlIter;

        ParticleVariable<Point> point;
        da->query(point, var, matl, patch, time);
        ParticleSubset* pset = point.getParticleSubset();
        ParticleVariable<long64> pid;
        da->query(pid, "p.particleID", matl, patch, time);
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
	     if (box.contains(point[*iter])) {
               cout << time << " " << patchIndex << " " << matl ;
               cout << " " << pid[*iter];
               cout << " " << point[*iter](0) 
                    << " " << point[*iter](1)
                    << " " << point[*iter](2) << endl;
	     }
          }
        }
      }
    } // end of patch loop
  } // end of level loop
}


/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


/*
 *  selectpart.cc: Get the range of particle ids in a given selection box
 *
 *  Written by:
 *   Biswajit Banerjee
 *   July 2005
 *
 *  Original code works only for the first timestep, and all the materials.
 *  Added -mat, -timesteplow, -timestephigh 
 *     options by Jonah Lee, December 2008
 *
 */

#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/SymmMatrix3.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/Array3.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

// declarations
void usage(const std::string& badarg, const std::string& progname);
void printParticleID(DataArchive* da, int mat, 
const bool tslow_set, const bool tsup_set, 
unsigned long & time_step_lower, unsigned long & time_step_upper,
const Box& box);

//borrowed from puda.h
void findTimestep_loopLimits( const bool tslow_set, 
                                const bool tsup_set,
                                const std::vector<double> times,
                                unsigned long & time_step_lower,
                                unsigned long & time_step_upper );

// Main
int main(int argc, char** argv)
{
  /*
   * Default values
   */
  double xmin = 0.0, ymin = 0.0, zmin = 0.0;
  double xmax = 1.0, ymax = 1.0, zmax = 1.0;
  string filebase;

  int mat = -1;
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = 1;
  unsigned long time_step_inc = 1;
  bool tslow_set = false;
  bool tsup_set = false;;

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
    } else if (s == "-mat") {
      mat = atoi(argv[++i]);
    } else if (s == "-timesteplow" ||
               s == "-timeStepLow" ||
               s == "-timestep_low") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
      tslow_set = true; 
    } else if (s == "-timestephigh" ||
               s == "-timeStepHigh" ||
               s == "-timestep_high") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
      tsup_set = true;
    } else if (s == "-timestepinc" ||
               s == "-timestepInc" ||
               s == "-timestep_inc") {
      time_step_inc = strtoul(argv[++i],(char**)NULL,10);
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
    printParticleID(da, mat, tslow_set, tsup_set, time_step_lower, 
         time_step_upper, box);

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

  cerr << "Usage: " << progname << "[options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -box <xmin> <ymin> <zmin> <xmax> <ymax> <zmax> (required)\n";
  cerr << "  -mat <material id>\n";
  cerr << "  -timesteplow <int>  (only outputs timestep from int)\n";
  cerr << "  -timestephigh <int> (only outputs timesteps upto int)\n";
  exit(1);
}

////////////////////////////////////////////////////////////////////////////
//
// Print particle IDs
//
////////////////////////////////////////////////////////////////////////////
void printParticleID(DataArchive* da, int mat, 
                    const bool tslow_set, 
                    const bool tsup_set,
                    unsigned long & time_step_lower,
                    unsigned long & time_step_upper,
                    const Box& box)
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
  // desired timesteps from the archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, 
    time_step_upper); 

  // Loop thru all time steps
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){

  double time = times[t];
  GridP grid = da->queryGrid(t);

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
      ConsecutiveRangeSet matls = da->queryMaterials(var, patch, t);
      ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
      for(; matlIter != matls.end(); matlIter++){
        int matl = *matlIter;
        if (mat != -1 && matl != mat) continue;

        ParticleVariable<Point> point;
        da->query(point, var, matl, patch, t);
        ParticleSubset* pset = point.getParticleSubset();
        ParticleVariable<long64> pid;
        da->query(pid, "p.particleID", matl, patch, t);
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
 } // end of time loop
}

void
findTimestep_loopLimits( const bool tslow_set, 
                                 const bool tsup_set,
                                 const vector<double> times,
                                 unsigned long & time_step_lower,
                                 unsigned long & time_step_upper )
{
  if( !tslow_set ) {
    time_step_lower = 0;
  }
  else if( time_step_lower >= times.size() ) {
    cerr << "timesteplow must be between 0 and " << times.size()-1 << "\n";
    abort();
  }
  if( !tsup_set ) {
    time_step_upper = times.size() - 1;
  }
  else if( time_step_upper >= times.size() ) {
    cerr << "timestephigh must be between 0 and " << times.size()-1 << "\n";
    abort();
  }
}



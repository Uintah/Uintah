/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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


// declarations
void usage(const std::string& badarg, const std::string& progname);
void printParticleID(Uintah::DataArchive* da, int mat, 
const bool tslow_set, const bool tsup_set, 
unsigned long & time_step_lower, unsigned long & time_step_upper,
const Uintah::Box& box);

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
  std::string filebase;

  int mat = -1;
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = 1;
  bool tslow_set = false;
  bool tsup_set = false;;

  // set defaults for std::cout
  std::cout.setf(std::ios::scientific,std::ios::floatfield);
  std::cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    std::string s=argv[i];
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
    } 
  }

  filebase = argv[argc-1];

  if(filebase == "" || filebase == argv[0]){
    std::cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    Uintah::DataArchive* da = scinew Uintah::DataArchive(filebase);

    // Setup box
    Uintah::Point lower(xmin,ymin,zmin);
    Uintah::Point upper(xmax,ymax,zmax);
    Uintah::Box box(lower, upper);
    
    // Get the particle IDs
    printParticleID(da, mat, tslow_set, tsup_set, time_step_lower, 
         time_step_upper, box);

  } catch (SCIRun::Exception& e) {
    std::cerr << "Caught exception: " << e.message() << std::endl;
    abort();
  } catch(...){
    std::cerr << "Caught unknown exception\n";
    abort();
  }
}

void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "") std::cerr << "Error parsing argument: " << badarg << std::endl;

  std::cerr << "Usage: " << progname << "[options] <archive file>\n\n";
  std::cerr << "Valid options are:\n";
  std::cerr << "  -box <xmin> <ymin> <zmin> <xmax> <ymax> <zmax> (required)\n";
  std::cerr << "  -mat <material id>\n";
  std::cerr << "  -timesteplow <int>  (only outputs timestep from int)\n";
  std::cerr << "  -timestephigh <int> (only outputs timesteps upto int)\n";
  exit(1);
}

////////////////////////////////////////////////////////////////////////////
//
// Print particle IDs
//
////////////////////////////////////////////////////////////////////////////
void printParticleID(Uintah::DataArchive* da, int mat, 
                    const bool tslow_set, 
                    const bool tsup_set,
                    unsigned long & time_step_lower,
                    unsigned long & time_step_upper,
                    const Uintah::Box& box)
{
  // Uintah::Box
  std::cerr << "Selction box = " << box << std::endl;

  // Check if the particle variable p.particleId and p.x are available
  std::vector<std::string> vars;
  std::vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == "p.particleID" || var == "p.x") variableFound = true;
  }
  if (!variableFound) {
    std::cerr << "p.particleID or p.x not found\n"; 
    exit(1);
  }

  // Now that the variable has been found, get the data for the
  // desired timesteps from the archive
  std::vector<int> index;
  std::vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, 
    time_step_upper); 

  // Loop thru all time steps
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){

  double time = times[t];
  Uintah::GridP grid = da->queryGrid(t);

  // Loop thru all the levels
  for(int l=0;l<grid->numLevels();l++){
    Uintah::LevelP level = grid->getLevel(l);

    // Loop thru all the patches
    Uintah::Level::const_patchIterator iter = level->patchesBegin(); 
    int patchIndex = 0;
    for(; iter != level->patchesEnd(); iter++){
      const Uintah::Patch* patch = *iter;
      ++patchIndex; 

      // Search for p.x
      std::string var = "p.x";

      // loop thru all the materials
      SCIRun::ConsecutiveRangeSet matls = da->queryMaterials(var, patch, t);
      SCIRun::ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
      for(; matlIter != matls.end(); matlIter++){
        int matl = *matlIter;
        if (mat != -1 && matl != mat) continue;

        Uintah::ParticleVariable<Uintah::Point> point;
        da->query(point, var, matl, patch, t);
        Uintah::ParticleSubset* pset = point.getParticleSubset();
        Uintah::ParticleVariable<Uintah::long64> pid;
        da->query(pid, "p.particleID", matl, patch, t);
        if(pset->numParticles() > 0){
          Uintah::ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
	     if (box.contains(point[*iter])) {
               std::cout << time << " " << patchIndex << " " << matl ;
               std::cout << " " << pid[*iter];
               std::cout << " " << point[*iter](0) 
                    << " " << point[*iter](1)
                    << " " << point[*iter](2) << std::endl;
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
                                 const std::vector<double> times,
                                 unsigned long & time_step_lower,
                                 unsigned long & time_step_upper )
{
  if( !tslow_set ) {
    time_step_lower = 0;
  }
  else if( time_step_lower >= times.size() ) {
    std::cerr << "timesteplow must be between 0 and " << times.size()-1 << "\n";
    abort();
  }
  if( !tsup_set ) {
    time_step_upper = times.size() - 1;
  }
  else if( time_step_upper >= times.size() ) {
    std::cerr << "timestephigh must be between 0 and " << times.size()-1 << "\n";
    abort();
  }
}



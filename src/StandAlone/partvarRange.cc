/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
 *  partvarRange.cc: Print out the range of values for a particle variable
 *
 *  Written by:
 *   Biswajit Banerjee
 */

#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Math/Matrix3.h>
#include <Core/DataArchive/DataArchive.h>
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

void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << " -mat <material id>\n";
  exit(1);
}

int main(int argc, char** argv)
{
  // Defaults
  int mat = 0;

  // Print out the usage and die
  if (argc <= 1) usage("", argv[0]);

  // Parse arguments
  string filebase;
  for (int i = 1; i < argc; i++) {
    string s = argv[i];
    if (s == "-mat") {
      mat = atoi(argv[++i]);
    }
  }
  filebase = argv[argc-1];
  if(filebase == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);

  try {
    DataArchive* da = scinew DataArchive(filebase);
    
    //______________________________________________________________________
    //              V A R S U M M A R Y   O P T I O N
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;
    da->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    //cout << "There are " << vars.size() << " variables:\n";
    //for(int i=0;i<(int)vars.size();i++) {
    //  cout << vars[i] << ": " << types[i]->getName() << endl;
    //}

      
    vector<int> index;
    vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    //cout << "There are " << index.size() << " timesteps:\n";
    //for(int i=0;i<(int)index.size();i++)
    //  cout << index[i] << ": " << times[i] << endl;
      
    // Var loop
    for(int v=0;v<(int)vars.size();v++){
      std::string var = vars[v];
      const Uintah::TypeDescription* td = types[v];
      const Uintah::TypeDescription* subtype = td->getSubType();

      // ParticleVariable switch
      switch(td->getType()){
      case Uintah::TypeDescription::ParticleVariable:

        switch(subtype->getType()){

        // Double variable
        case Uintah::TypeDescription::double_type:
          {
            // Set up variables to store min and max
            double min = 1.0e30, max = -1.0e30;

            // Time loop
            unsigned long time_step_lower = 0;
            unsigned long time_step_upper = times.size()-1;
            unsigned long t=time_step_lower;
            for(; t<=time_step_upper; t++){
              GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                LevelP level = grid->getLevel(l);

                // Patch loop
                Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Patch* patch = *pIter;
                  ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    ParticleVariable<double> value;
                    da->query(value, var, matl, patch, t);
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=Min(min, value[*iter]);
                        max=Max(max, value[*iter]);
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            cout << var << " min = " << min << " max = " << max << endl;
          } // end double case
        break;

        // Float variable
        case Uintah::TypeDescription::float_type:
          {
            // Set up variables to store min and max
            float min = 1.0e20, max = -1.0e20;

            // Time loop
            unsigned long time_step_lower = 0;
            unsigned long time_step_upper = times.size()-1;
            unsigned long t=time_step_lower;
            for(; t<=time_step_upper; t++){
              GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                LevelP level = grid->getLevel(l);

                // Patch loop
                Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Patch* patch = *pIter;
                  ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    ParticleVariable<float> value;
                    da->query(value, var, matl, patch, t);
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=Min(min, value[*iter]);
                        max=Max(max, value[*iter]);
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            cout << var << " min = " << min << " max = " << max << endl;
          } // end float case
        break;

        // Int variable
        case Uintah::TypeDescription::int_type:
          {
            // Set up variables to store min and max
            int min = 40000000, max = -40000000;

            // Time loop
            unsigned long time_step_lower = 0;
            unsigned long time_step_upper = times.size()-1;
            unsigned long t=time_step_lower;
            for(; t<=time_step_upper; t++){
              GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                LevelP level = grid->getLevel(l);

                // Patch loop
                Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Patch* patch = *pIter;
                  ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    ParticleVariable<int> value;
                    da->query(value, var, matl, patch, t);
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=Min(min, value[*iter]);
                        max=Max(max, value[*iter]);
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            cout << var << " min = " << min << " max = " << max << endl;
          } // end int case
        break;

        // Vector variable
        case Uintah::TypeDescription::Vector:
          {
            // Set up variables to store min and max
            double min = 1.0e30, max = -1.0e30;

            // Time loop
            unsigned long time_step_lower = 0;
            unsigned long time_step_upper = times.size()-1;
            unsigned long t=time_step_lower;
            for(; t<=time_step_upper; t++){
              GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                LevelP level = grid->getLevel(l);

                // Patch loop
                Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Patch* patch = *pIter;
                  ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    ParticleVariable<Vector> value;
                    da->query(value, var, matl, patch, t);
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=Min(min, value[*iter].length2());
                        max=Max(max, value[*iter].length2());
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            cout << var << " min = " << sqrt(min) 
                 << " max = " << sqrt(max) << endl;
          } // end Vector case
        break;

        // Matrix3 variable
        case Uintah::TypeDescription::Matrix3:
          {
            // Set up variables to store min and max
            double min = 1.0e30, max = -1.0e30;

            // Time loop
            unsigned long time_step_lower = 0;
            unsigned long time_step_upper = times.size()-1;
            unsigned long t=time_step_lower;
            for(; t<=time_step_upper; t++){
              GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                LevelP level = grid->getLevel(l);

                // Patch loop
                Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Patch* patch = *pIter;
                  ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    ParticleVariable<Matrix3> value;
                    da->query(value, var, matl, patch, t);
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=Min(min, value[*iter].NormSquared());
                        max=Max(max, value[*iter].NormSquared());
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            cout << var << " min = " << sqrt(min) 
                 << " max = " << sqrt(max) << endl;
          } // end Matrix3 case
        break;

        case Uintah::TypeDescription::Point:
        break;

        case Uintah::TypeDescription::long64_type:
        break;

        default:
          {
            cerr << "Particle Variable of unknown type: " 
                 << subtype->getName() << endl;
          }
        break;

        } // end switch subtype

      case Uintah::TypeDescription::NCVariable:
      break;

      case Uintah::TypeDescription::CCVariable:
      break;

      case Uintah::TypeDescription::SFCXVariable:
      break;

      case Uintah::TypeDescription::SFCYVariable:
      break;

      case Uintah::TypeDescription::SFCZVariable:
      break;

      default:
      break;

      } // end switch type
    
    } // end var loop
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}


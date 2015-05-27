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


void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "")
    std::cerr << "Error parsing argument: " << badarg << std::endl;
  std::cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  std::cerr << "Valid options are:\n";
  std::cerr << " -mat <material id>\n";
  exit(1);
}

int main(int argc, char** argv)
{
  // Defaults
  int mat = 0;

  // Print out the usage and die
  if (argc <= 1) usage("", argv[0]);

  // Parse arguments
  std::string filebase;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "-mat") {
      mat = atoi(argv[++i]);
    }
  }
  filebase = argv[argc-1];
  if(filebase == ""){
    std::cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  // set defaults for std::cout
  std::cout.setf(std::ios::scientific,std::ios::floatfield);
  std::cout.precision(8);

  try {
    Uintah::DataArchive* da = scinew Uintah::DataArchive(filebase);
    
    //______________________________________________________________________
    //              V A R S U M M A R Y   O P T I O N
    std::vector<std::string> vars;
    std::vector<const Uintah::TypeDescription*> types;
    da->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    //std::cout << "There are " << vars.size() << " variables:\n";
    //for(int i=0;i<(int)vars.size();i++) {
    //  std::cout << vars[i] << ": " << types[i]->getName() << std::endl;
    //}

      
    std::vector<int> index;
    std::vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    //std::cout << "There are " << index.size() << " timesteps:\n";
    //for(int i=0;i<(int)index.size();i++)
    //  std::cout << index[i] << ": " << times[i] << std::endl;
      
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
              Uintah::GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                Uintah::LevelP level = grid->getLevel(l);

                // Patch loop
                Uintah::Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Uintah::Patch* patch = *pIter;
                  SCIRun::ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  SCIRun::ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    Uintah::ParticleVariable<double> value;
                    da->query(value, var, matl, patch, t);
                    Uintah::ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      Uintah::ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=std::min(min, value[*iter]);
                        max=std::max(max, value[*iter]);
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            std::cout << var << " min = " << min << " max = " << max << std::endl;
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
              Uintah::GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                Uintah::LevelP level = grid->getLevel(l);

                // Patch loop
                Uintah::Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Uintah::Patch* patch = *pIter;
                  SCIRun::ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  SCIRun::ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    Uintah::ParticleVariable<float> value;
                    da->query(value, var, matl, patch, t);
                    Uintah::ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      Uintah::ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=std::min(min, value[*iter]);
                        max=std::max(max, value[*iter]);
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            std::cout << var << " min = " << min << " max = " << max << std::endl;
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
              Uintah::GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                Uintah::LevelP level = grid->getLevel(l);

                // Patch loop
                Uintah::Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Uintah::Patch* patch = *pIter;
                  SCIRun::ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  SCIRun::ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    Uintah::ParticleVariable<int> value;
                    da->query(value, var, matl, patch, t);
                    Uintah::ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      Uintah::ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=std::min(min, value[*iter]);
                        max=std::max(max, value[*iter]);
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            std::cout << var << " min = " << min << " max = " << max << std::endl;
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
              Uintah::GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                Uintah::LevelP level = grid->getLevel(l);

                // Patch loop
                Uintah::Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Uintah::Patch* patch = *pIter;
                  SCIRun::ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  SCIRun::ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    Uintah::ParticleVariable<Uintah::Vector> value;
                    da->query(value, var, matl, patch, t);
                    Uintah::ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      Uintah::ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=std::min(min, value[*iter].length2());
                        max=std::max(max, value[*iter].length2());
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            std::cout << var << " min = " << sqrt(min) 
                 << " max = " << sqrt(max) << std::endl;
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
              Uintah::GridP grid = da->queryGrid(t);

              // Level loop
              for(int l=0;l<grid->numLevels();l++){
                Uintah::LevelP level = grid->getLevel(l);

                // Patch loop
                Uintah::Level::const_patchIterator pIter = level->patchesBegin();
                for(; pIter != level->patchesEnd(); pIter++){
                  const Uintah::Patch* patch = *pIter;
                  SCIRun::ConsecutiveRangeSet matls = 
                    da->queryMaterials(var, patch, t);

                  // Material loop
                  SCIRun::ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  for(; matlIter != matls.end(); matlIter++){
                    int matl = *matlIter;

                    if (matl != mat) continue;

                    Uintah::ParticleVariable<Uintah::Matrix3> value;
                    da->query(value, var, matl, patch, t);
                    Uintah::ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      Uintah::ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        min=std::min(min, value[*iter].NormSquared());
                        max=std::max(max, value[*iter].NormSquared());
                      }
                    }
                  } // end material loop
                } // end patch loop
              } // end level loop
            } // end time loop
            std::cout << var << " min = " << sqrt(min) 
                 << " max = " << sqrt(max) << std::endl;
          } // end Matrix3 case
        break;

        case Uintah::TypeDescription::Point:
        break;

        case Uintah::TypeDescription::long64_type:
        break;

        default:
          {
            std::cerr << "Particle Variable of unknown type: " 
                 << subtype->getName() << std::endl;
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
  } catch (SCIRun::Exception& e) {
    std::cerr << "Caught exception: " << e.message() << std::endl;
    abort();
  } catch(...){
    std::cerr << "Caught unknown exception\n";
    abort();
  }
}


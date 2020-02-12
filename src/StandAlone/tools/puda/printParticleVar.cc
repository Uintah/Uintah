/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#include <StandAlone/tools/puda/printParticleVar.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <string>
#include <vector>

using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////////
//
// Print ParticleVariable
//
void
Uintah::printParticleVariable( DataArchive      * da,
                               CommandLineFlags & clf,
                               int mat)
{
  // Check if the particle variable is available
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables( vars, num_matls, types );
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == clf.particleVariable) variableFound = true;
  }
  if (!variableFound) {
    cerr << "Variable " << clf.particleVariable << " not found\n";
    exit(1);
  }

  // Now that the variable has been found, get the data for all
  // available time steps // from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  //cout << "There are " << index.size() << " timesteps:\n";


  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
  
  bool useParticleID = true;

  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t++){
    double time = times[t];
    //cout << "Time = " << time << "\n";
    GridP grid = da->queryGrid(t);

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patch_iterator iter = level->patchesBegin();
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
            ConsecutiveRangeSet matls = da->queryMaterials(var, patch, t);
            ConsecutiveRangeSet::iterator matlIter = matls.begin();
            for(; matlIter != matls.end(); matlIter++){
              int matl = *matlIter;
              if (mat != -1 && matl != mat) continue;

              // Find the name of the variable
              if (var == clf.particleVariable) {
                //cout << "Material: " << matl << "\n";
                switch(subtype->getType()){
                case Uintah::TypeDescription::double_type:
                  {
                    ParticleVariable<double> value;
                    da->query( value, var, matl, patch, t) ;
                    ParticleVariable<long64> pid;
                    if( useParticleID ) {
                      try {
                        // If particleID wasn't saved, just move on...
                        da->query( pid, "p.particleID", matl, patch, t );
                      } catch( Exception & e ) {
                        useParticleID = false;
                      }
                    }
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl;
                        if( useParticleID ) {
                          cout << " " << pid[*iter];
                        }
                        cout << " " << value[*iter] << "\n";
                      }
                    }
                  }
                break;
                case Uintah::TypeDescription::float_type:
                  {
                    ParticleVariable<float> value;
                    da->query( value, var, matl, patch, t );
                    ParticleVariable<long64> pid;
                    if( useParticleID ) {
                      try {
                        // If particleID wasn't saved, just move on...
                        da->query( pid, "p.particleID", matl, patch, t );
                      }
                      catch( Exception & e ) {
                        useParticleID = false;
                      }
                    }
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
                        if( useParticleID ) {
                          cout << " " << pid[*iter];
                        }
                        cout << " " << value[*iter] << "\n";
                      }
                    }
                  }
                break;
                case Uintah::TypeDescription::int_type:
                  {
                    ParticleVariable<int> value;
                    da->query( value, var, matl, patch, t );
                    ParticleSubset* pset = value.getParticleSubset();
                    ParticleVariable<long64> pid;
                    if( useParticleID ) {
                      try {
                        // If particleID wasn't saved, just move on...
                        da->query( pid, "p.particleID", matl, patch, t );
                      }
                      catch( Exception & e ) {
                        useParticleID = false;
                      }
                    }
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl;
                        if( useParticleID ) {
                          cout << " " << pid[*iter];
                        }
                        cout << " " << value[*iter] << "\n";
                      }
                    }
                  }
                break;
                case Uintah::TypeDescription::Point:
                  {
                    ParticleVariable<Point> value;
                    da->query( value, var, matl, patch, t );
                    ParticleSubset* pset = value.getParticleSubset();
                    ParticleVariable<long64> pid;
                    if( useParticleID ) {
                      try {
                        // If particleID wasn't saved, just move on...
                        da->query( pid, "p.particleID", matl, patch, t );
                      }
                      catch( Exception & e ) {
                        useParticleID = false;
                      }
                    }
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
                        if( useParticleID ) {
                          cout << " " << pid[*iter];
                        }
                        cout << " " << value[*iter](0)
                             << " " << value[*iter](1)
                             << " " << value[*iter](2) << "\n";
                      }
                    }
                  }
                break;
                case Uintah::TypeDescription::Vector:
                  {
                    ParticleVariable<Vector> value;
                    da->query( value, var, matl, patch, t );
                    ParticleVariable<long64> pid;
                    if( useParticleID ) {
                      try {
                        // If particleID wasn't saved, just move on...
                        da->query( pid, "p.particleID", matl, patch, t );
                      }
                      catch( Exception & e ) {
                        useParticleID = false;
                      }
                    }
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        if( useParticleID ) {
                          cout << time << " " << patchIndex << " " << matl ;
                        }
                        cout << " " << pid[*iter];
                        cout << " " << value[*iter][0]
                             << " " << value[*iter][1]
                             << " " << value[*iter][2] << "\n";
                      }
                    }
                  }
                break;
                case Uintah::TypeDescription::Matrix3:
                  {
                    ParticleVariable<Matrix3> value;
                    da->query( value, var, matl, patch, t );
                    ParticleVariable<long64> pid;
                    if( useParticleID ) {
                      try {
                        // If particleID wasn't saved, just move on...
                        da->query( pid, "p.particleID", matl, patch, t );
                      }
                      catch( Exception & e ) {
                        useParticleID = false;
                      }
                    }
                    ParticleSubset* pset = value.getParticleSubset();
                    if(pset->numParticles() > 0){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
                        if( useParticleID ) {
                          cout << " " << pid[*iter];
                        }
                        for (int ii = 0; ii < 3; ++ii) {
                          for (int jj = 0; jj < 3; ++jj) {
                            cout << " " << value[*iter](ii,jj) ;
                          }
                        }
                        cout << "\n";
                      }
                    }
                  }
                break;
                case Uintah::TypeDescription::long64_type:
                  {
                    ParticleVariable<long64> value;
                    da->query( value, var, matl, patch, t );
                    ParticleSubset* pset = value.getParticleSubset();
                    if( pset->numParticles() > 0 ){
                      ParticleSubset::iterator iter = pset->begin();
                      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl;
                        cout << " " << value[*iter] << "\n";
                      }
                    }
                  }
                break;
                default:
                  cerr << "Particle Variable of unknown type: "
                       << subtype->getType() << "\n";
                  break;
                }
              } // end of var compare if
            } // end of material loop
          } // end of ParticleVariable if
        } // end of variable loop
      } // end of patch loop
    } // end of level loop
  } // end of time step loop
} // end printParticleVariable()

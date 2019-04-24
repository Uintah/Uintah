/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <StandAlone/tools/puda/aveParticleQ.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
//  Compute the mass weight averages for each timestep

void
Uintah::aveParticleQuanties( DataArchive      * da, 
                             CommandLineFlags & clf )
{
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  
  cout << "There are " << vars.size() << " variables:\n";
  
  for(int i=0;i< (int)vars.size(); i++) {
    cout << vars[i] << ": " << types[i]->getName() << endl;
  }
      
  vector<int>     index;
  vector<double>  times;
  da->queryTimesteps(index, times);
  cout << "There are " << index.size() << " timesteps:\n";
  
  ASSERTEQ(index.size(), times.size());
  
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << index[i] << ": " << times[i] << endl;
  }
      
  //__________________________________
  //  Open file
  ostringstream fnum;
  string filename( "aveParticleQ.dat" );
  ofstream outfile( filename.c_str() ); 
  outfile.setf(ios::scientific,ios::floatfield);
  outfile.precision(15);
  outfile << "# time                 meanVel.x            meanVel.y             meanVel.z             totalMass              KE " << endl;
  //__________________________________
  //  Loop over timesteps
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);

  for(unsigned long t=clf.time_step_lower; t<=clf.time_step_upper; t+=clf.time_step_inc){
    double time = times[t];
    cout << "time = " << time << endl;
    
    GridP grid = da->queryGrid(t);
    LevelP level = grid->getLevel(grid->numLevels()-1);
    cout << "Level: " << grid->numLevels() - 1 <<  endl;

    Vector total_mom(0.,0.,0.);
    double KE        = 0.;
    double total_mass=0.;
    
    //__________________________________
    //  Loop over patches
    for(Level::const_patch_iterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
      
      int matl = clf.matl;
      //__________________________________
      //  retrieve variables
      ParticleVariable<Point>  pPos;
      ParticleVariable<Vector> pVel;
      ParticleVariable<double> pMass;
      
      da->query( pPos, "p.x",        matl, patch, t );
      da->query( pVel, "p.velocity", matl, patch, t );
      da->query( pMass,"p.mass",     matl, patch, t );

      ParticleSubset* pset = pPos.getParticleSubset();

      if(pset->numParticles() > 0){
      
        //__________________________________
        //  Compute sums 
        ParticleSubset::iterator piter = pset->begin();
        
        for(;piter != pset->end(); piter++){
          particleIndex idx = *piter;
          
          double mass       = pMass[idx];
          double vel_mag_sq = pVel[idx].length2();

          total_mass += mass;
          total_mom  += mass * pVel[idx]; 
          KE         += 0.5 * mass * vel_mag_sq;
          
        } // for
      }  // if
    }  // for patches
    
    Vector mean_vel = total_mom/total_mass;
   

   outfile << time << " " << mean_vel.x() << " " << mean_vel.y() << " " << mean_vel.z() <<" " << total_mass << " " << KE << endl; 
  }
} // end


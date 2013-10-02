/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <StandAlone/tools/puda/jim2.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              J I M 2   O P T I O N
// Reads in particle mass and velocity, and compiles mean velocity magnitude
// and total kinetic energy for all particles of a specified material within
// the range of timesteps specified

void
Uintah::jim2( DataArchive * da, CommandLineFlags & clf )
{
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  cout << "There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++)
    cout << vars[i] << ": " << types[i]->getName() << endl;
      
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << index[i] << ": " << times[i] << endl;
  }
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);

  ostringstream fnum;
  string filename("time_meanvel_KE.dat");
  ofstream outfile(filename.c_str());

  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
    double time = times[t];
    cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);

    Vector mean_vel(0.,0.,0.);
    double KE = 0.;
    double total_mass=0.;
      LevelP level = grid->getLevel(grid->numLevels()-1);
      cout << "Level: " << grid->numLevels() - 1 <<  endl;
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        int matl = clf.matl_jim;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<Point> value_pos;
        ParticleVariable<Vector> value_vel;
        ParticleVariable<double> value_mass;
        da->query(value_pos, "p.x",        matl, patch, t);
        da->query(value_vel, "p.velocity", matl, patch, t);
        da->query(value_mass,"p.mass",     matl, patch, t);

        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator piter = pset->begin();
          for(;piter != pset->end(); piter++){
            double vel_mag = value_vel[*piter].length();
            mean_vel+=value_vel[*piter]*value_mass[*piter];
            KE+=value_mass[*piter]*vel_mag*vel_mag;
            total_mass+=value_mass[*piter];
          } // for
        }  //if
      }  // for patches
    mean_vel/=total_mass;
    double mean_vel_mag = mean_vel.length();
    KE*=.5;

   outfile.precision(15);
   outfile << time << " " << mean_vel_mag << " " << total_mass << " " << KE << endl; 

  }
} // end jim2()


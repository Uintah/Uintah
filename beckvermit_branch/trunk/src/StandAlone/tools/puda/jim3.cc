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


#include <StandAlone/tools/puda/jim3.h>

#include <StandAlone/tools/puda/util.h>

#include <Core/DataArchive/DataArchive.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              J I M 1   O P T I O N
// This currently pulls out particle position, velocity and ID
// and prints that on one line for each particle.  This is useful
// for postprocessing of particle data for particles which move
// across patches.

void
Uintah::jim3( DataArchive * da, CommandLineFlags & clf )
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
      
  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
    double time = times[t];
    cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);
    ostringstream fnum;
    string filename;
    fnum << setw(4) << setfill('0') << t/clf.time_step_inc;
    string partroot("hist.out");
    filename = partroot+ fnum.str();
    ofstream partfile(filename.c_str());
    double stress_bins[100];
    int num_in_bin[100];
    for(int i=0;i<100;i++){
      num_in_bin[i]=0;
    }

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      double min_eq_stress= 1e15;
      double max_eq_stress=-1e15;
//      cout << "Level: " <<  endl;
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        int matl = clf.matl_jim;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<Point> value_pos;
        ParticleVariable<Matrix3> value_strs;
        da->query(value_pos, "p.x",          matl, patch, t);
        da->query(value_strs,"p.stress",     matl, patch, t);
        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            Matrix3 one; one.Identity();
            Matrix3 Mdev = value_strs[*iter] - one*(value_strs[*iter].Trace()/3.0);
            double eq_stress=sqrt(Mdev.NormSquared()*1.5);
            min_eq_stress = min(min_eq_stress,eq_stress);
            max_eq_stress = max(max_eq_stress,eq_stress);
          } // for
        }  //if
      }  // for patches

      double stress_interval = (max_eq_stress - min_eq_stress)/100.;
      for(int i=0;i<100;i++){
        stress_bins[i]=min_eq_stress+((double) i)*stress_interval;
      }

      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        int matl = clf.matl_jim;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<Point> value_pos;
        ParticleVariable<Matrix3> value_strs;
        da->query(value_pos, "p.x",          matl, patch, t);
        da->query(value_strs,"p.stress",     matl, patch, t);
        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            Matrix3 one; one.Identity();
            Matrix3 Mdev = value_strs[*iter] - one*(value_strs[*iter].Trace()/3.0);
            double eq_stress=sqrt(Mdev.NormSquared()*1.5);
            int bin = ((int) ((eq_stress-min_eq_stress)/stress_interval));
            num_in_bin[bin]++;
          } // for
        }  //if
      }  // for patches
    }   // for levels
    for(int i=0;i<100;i++){
      partfile << stress_bins[i] << " " << num_in_bin[i] << endl;
    }
  }
} // end jim3()

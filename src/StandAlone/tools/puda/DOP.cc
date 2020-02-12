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

#include <StandAlone/tools/puda/DOP.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              DOP   O P T I O N
// Finds the location of the particle of the specified material farthest
// in the direction specified in the -dir option, where:
// -1 -> -x
//  1 -> +x
// -2 -> -y
//  2 -> +y
// -3 -> -z
//  3 -> +z

void
Uintah::DOP( DataArchive * da, CommandLineFlags & clf )
{
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables( vars, num_matls, types );
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
  string filename("Time_DOP.dat");
  ofstream outfile(filename.c_str());

  int dir = clf.dir;
  double comp = (double) abs(dir);
  double sign = (double) (dir/comp);
  double ddir = (double) dir;
  for(unsigned long t=clf.time_step_lower;
                    t<=clf.time_step_upper;
                    t+=clf.time_step_inc){
    double time = times[t];
    cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);

    double maxDOP = -9.e99;
    double minDOP =  9.e99;
    LevelP level = grid->getLevel(grid->numLevels()-1);
    for(Level::const_patch_iterator iter = level->patchesBegin();
        iter != level->patchesEnd(); iter++){
      const Patch* patch = *iter;
      int matl = clf.matl;
      //__________________________________
      //   P A R T I C L E   V A R I A B L E
      ParticleVariable<Point> value_pos;
      da->query(value_pos, "p.x",        matl, patch, t);

      ParticleSubset* pset = value_pos.getParticleSubset();
      if(pset->numParticles() > 0){
        ParticleSubset::iterator piter = pset->begin();
        if(sign>0){
          for(;piter != pset->end(); piter++){
            maxDOP=max(maxDOP,value_pos[*piter](comp-1));
          } // for
        } else {
          for(;piter != pset->end(); piter++){
            minDOP=min(minDOP,value_pos[*piter](comp-1));
          } // for
        }
      }  //if
    }  // for patches

   outfile.precision(12);
   if(sign>0){
     outfile << time << " " << maxDOP << endl; 
   } else {
     outfile << time << " " << minDOP << endl; 
   }

  }
} // end DOP()


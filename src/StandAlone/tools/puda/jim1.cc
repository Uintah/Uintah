/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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


#include <StandAlone/tools/puda/jim1.h>

#include <StandAlone/tools/puda/util.h>

#include <Core/DataArchive/DataArchive.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              J I M 1   O P T I O N
// This currently sums up the number of particles in the column of cells
// in each cell in the x-y plane.
// With gnuplot, one can plot the results to see how well the particles are
// distributed using:

// gnuplot> set dgrid hix-1, hiy-1
// gnuplot> set hidden3d
// gnuplot> splot "partout0000" using 1:2:3 w lines

// In the above, hix and hiy are the x and y components, respectively, of
// the "hi" value reported when you run this script.
// Usage, when run from inside an uda, looks like, e.g.; >puda -jim1 -matl 33 .
// where the -matl flag indicates the maximum material number you want included
// in your sum (i.e. one may wish to exclude piston materials).  That is,
// sum up the particles for materials m, where: 0 < m < matl.

void
Uintah::jim1( DataArchive * da, CommandLineFlags & clf )
{
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables( vars, num_matls, types );
  ASSERTEQ(vars.size(), types.size());
  cout << "#There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++){
    cout << vars[i] << ": " << types[i]->getName() << endl;
  }

  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "#There are " << index.size() << " timesteps:\n";
//  for( int i = 0; i < (int)index.size(); i++ ) {
//    cout << index[i] << ": " << times[i] << endl;
//  }
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, 
                           clf.time_step_lower, clf.time_step_upper);
      
  for(unsigned long t= clf.time_step_lower;
                    t<=clf.time_step_upper;
                    t+=clf.time_step_inc){
    // double time = times[t];
    //cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);
    LevelP level = grid->getLevel(0);
    IntVector low, hi;
    level->findNodeIndexRange(low, hi);
    cout << "low = " << low << endl;
    cout << "hi = " << hi << endl;
    ostringstream fnum;
    string filename;
    fnum << setw(4) << setfill('0') << t/clf.time_step_inc;
    string partroot("partout");
    filename = partroot+ fnum.str();
    ofstream partfile(filename.c_str());
    // I know the following is suspect, if it starts acting funny,
    // use STL vectors
    int cell_bins[hi.x()-1][hi.y()-1];
    for(int j=0;j<hi.y()-1;j++){
      for(int i=0;i<hi.x()-1;i++){
        cell_bins[i][j]=0;
      }
    }

    for(int matl=0; matl < clf.matl; matl++){
     cout << "matl = " << matl << endl;
      for(Level::const_patch_iterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<Point> px;
        da->query(px, "p.x",          matl, patch, t);
        ParticleSubset* pset = px.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            IntVector cI;
            patch->findCell(px[*iter],cI);
            cell_bins[cI.x()][cI.y()]++;
          } // for
        }  //if
      }  // for patches
    }   // for materials
    for(int j=0;j<hi.y()-1;j++){
      for(int i=0;i<hi.x()-1;i++){
        partfile << i << " " << j << " " << cell_bins[i][j] << endl;
      }
    }
  }
} // end jim1()

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


#include <StandAlone/tools/puda/PIC.h>

#include <StandAlone/tools/puda/util.h>

#include <Core/DataArchive/DataArchive.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              P I C   O P T I O N
//  Takes a cell index and prints all the particle ids that live in that 
//  cell for each timestep.

void
Uintah::PIC( DataArchive * da, CommandLineFlags & clf, int cx, int cy, int cz )
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
    fnum << cx << "." << cy << "." << cz << ".time." << setw(4) << setfill('0') << t/clf.time_step_inc;
    string partroot("partIds.index.");
    filename = partroot + fnum.str();
    ofstream partfile(filename.c_str());

    // Cell to find
    IntVector cell(cx,cy,cz);

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Find the patch on which the cell lives
      // don't include extra cells -> false
      const Patch * patch = level->getPatchFromIndex(cell, true);
      if(patch != NULL)
      {
        cout << "Cell found on patch: " << patch->getID() << " on level: " << level << endl;   
      } else {
         continue;
      }

      int matl = clf.matl_jim;
      //__________________________________
      //   P A R T I C L E   V A R I A B L E
      ParticleVariable<long64> value_pID;
      ParticleVariable<Point> value_pos;
      ParticleVariable<Vector> value_vel;
      ParticleVariable<Matrix3> value_strs;
      da->query(value_pID, "p.particleID", matl, patch, t);
      da->query(value_pos, "p.x",          matl, patch, t);
      ParticleSubset* pset = value_pos.getParticleSubset();
      if(pset->numParticles() > 0){
        ParticleSubset::iterator iter = pset->begin();
        for(;iter != pset->end(); iter++){
          IntVector ci;
          patch->findCell(value_pos[*iter], ci);

          if(ci.x() == cell.x() && ci.y() == cell.y() && ci.z() == cell.z())
          {
            partfile << value_pID[*iter] << endl;
          }
        } // for
      }  //if
    }   // for levels
  }
} // end PIC()

/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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


#include <StandAlone/tools/puda/geocosmtets.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////

void
Uintah::geocosmtets( DataArchive * da, CommandLineFlags & clf )
{
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, num_matls, types);
  ASSERTEQ(vars.size(), types.size());
  cout << "#There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++)
    cout << vars[i] << ": " << types[i]->getName() << endl;
      
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "#There are " << index.size() << " timesteps:\n";
//  for( int i = 0; i < (int)index.size(); i++ ) {
//    cout << index[i] << ": " << times[i] << endl;
//  }
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
      
  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
   int maxMatl = clf.matl;
   for(int matl=0; matl<=maxMatl; matl++){
    //double time = times[t];
    //cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);
    ostringstream fnum, mnum;
    string filename;
    fnum << setw(3) << setfill('0') << t/clf.time_step_inc;
    mnum << setw(3) << setfill('0') << matl;
    string partroot("partTet");
    filename = partroot + "." + fnum.str() + "." + mnum.str();
    ofstream partfile(filename.c_str());

    partfile << "# x y z n0x n0y n0z n1x n1y n1z n2x n2y n2z n3x n3y n3z pID color sigxx, sigyy sigzz sigyz sigxz sigxy pressure equiv_stress plasStrain" << endl;

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      for(Level::const_patch_iterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
       const Patch* patch = *iter;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<long64> pID;
        ParticleVariable<Point>  pos;
        ParticleVariable<Vector> vel;
        ParticleVariable<double> plas, col;
        ParticleVariable<Matrix3> stress, scalefac;
        da->query(pos,     "p.x",              matl, patch, t);
        da->query(pID,     "p.particleID",     matl, patch, t);
        da->query(col,     "p.color",          matl, patch, t);
        da->query(plas,    "p.plasticStrain",  matl, patch, t);
        da->query(stress,  "p.stress",         matl, patch, t);
        da->query(scalefac,"p.scalefactor",    matl, patch, t);
//      Note, calling this dsize, just to make it easy to copy/paste
//      code from cptiInterpolator, but dsize HERE, which is actually
//      p.scalefactor, has had a factor of the cell size multiplied in.
//      Thus, it is not normalized to the cell size, but refers to absolute
//      dimensions.
        ParticleSubset* pset = pos.getParticleSubset();
        Point node_loc[4];
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            Matrix3 dsize=scalefac[*iter];
            node_loc[0] = pos[*iter] +
                          Vector(-dsize(0,0)-dsize(0,1)-dsize(0,2),
                                 -dsize(1,0)-dsize(1,1)-dsize(1,2),
                                 -dsize(2,0)-dsize(2,1)-dsize(2,2))*0.25;
            node_loc[1]=node_loc[0] +Vector(dsize(0,0),dsize(1,0),dsize(2,0));
            node_loc[2]=node_loc[0] +Vector(dsize(0,1),dsize(1,1),dsize(2,1));
            node_loc[3]=node_loc[0] +Vector(dsize(0,2),dsize(1,2),dsize(2,2));
            Matrix3 sig = stress[*iter];
            double pressure = (-1.0/3.0)*sig.Trace();
            double eqStress = sqrt(0.5*((sig(0,0)-sig(1,1))*(sig(0,0)-sig(1,1))
                            +           (sig(1,1)-sig(2,2))*(sig(1,1)-sig(2,2))
                            +           (sig(2,2)-sig(0,0))*(sig(2,2)-sig(0,0))
                            +       6.0*(sig(0,1)*sig(0,1) + sig(1,2)*sig(1,2) +
                                        sig(2,0)*sig(2,0))));

            partfile << pos[*iter].x() << " " << pos[*iter].y() << " " << pos[*iter].z() << " " 
                     << node_loc[0].x() << " " << node_loc[0].y() << " " << node_loc[0].z() << " "
                     << node_loc[1].x() << " " << node_loc[1].y() << " " << node_loc[1].z() << " "
                     << node_loc[2].x() << " " << node_loc[2].y() << " " << node_loc[2].z() << " "
                     << node_loc[3].x() << " " << node_loc[3].y() << " " << node_loc[3].z() << " "
                     << pID[*iter] << " " 
                     << col[*iter]         << " " << sig(0,0) << " " 
                     << sig(1,1) << " " << sig(2,2) << " " 
                     << sig(1,2) << " " << sig(0,2) << " "
                     << sig(0,1) << " " << pressure << " " 
                     << eqStress << " " << plas[*iter] << endl;
          } // for
        }  //if
      }  // for patches
    }   // for levels
   }    // for matls
  }     // for timesteps
} // end geocosmtets()

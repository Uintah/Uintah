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


#include <StandAlone/tools/puda/geocosm.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

// USING THE puda -geocosm OPTION

// This code exists to extract select data from all particles,
// write it out in a series of text files that, named, e.g. p.001.007,
// where, in that example, 001 refers to output step 1, and 007 refers to
// data for material (aka, "group") 7.

// To invoke:

// > /path/to/puda/puda -geocosm -matl M -timesteplow TL -timestephigh TH MyUda.000

// This will extract data for all materials up to and including M-1 (materials
// are numbered from 0), starting at timestep TL up to timestep TH, from 
// MyUda.000
// You may find it easier to cd into MyUda.000, and then run the command as:

// > /path/to/puda/puda -geocosm -matl M -timesteplow TL -timestephigh TH .

// where the "." at the end indicates that the current directory will be worked
// on.  This leaves all of the output files inside the uda directory from 
// which they were extracted.

////////////////////////////////////////////////////////////////////////

void
Uintah::geocosm( DataArchive * da, CommandLineFlags & clf )
{
  bool have_volume = false;
  bool have_plastic_strain = false;
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, num_matls, types);
  ASSERTEQ(vars.size(), types.size());
  cout << "#There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++){
    cout << vars[i] << ": " << types[i]->getName() << endl;
    if(vars[i]=="p.volume"){
       have_volume=true;
    }
    if(vars[i]=="p.plasticStrain"){
       have_plastic_strain=true;
    }
  }

  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "#There are " << index.size() << " timesteps:\n";

  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
      
  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
   int maxMatl = clf.matl;
   for(int matl=0; matl<maxMatl; matl++){
    //double time = times[t];
    //cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);
    ostringstream fnum, mnum;
    string filename;
    fnum << setw(3) << setfill('0') << t/clf.time_step_inc;
    mnum << setw(3) << setfill('0') << matl;
    string partroot("partout");
    filename = partroot + "." + fnum.str() + "." + mnum.str();
    ofstream partfile(filename.c_str());

    partfile.precision(12);

    if(have_volume && have_plastic_strain){
      partfile << "# x y z pID color sigxx, sigyy sigzz sigyz sigxz sigxy pressure equiv_stress plasStrain volume" << endl;
    }else if(!have_volume && have_plastic_strain){
      partfile << "# x y z pID color sigxx, sigyy sigzz sigyz sigxz sigxy pressure equiv_stress plasStrain" << endl;
    }else if(have_volume && !have_plastic_strain){
      partfile << "# x y z pID color sigxx, sigyy sigzz sigyz sigxz sigxy pressure equiv_stress volume" << endl;
    }else if(!have_volume && !have_plastic_strain){
      partfile << "# x y z pID color sigxx, sigyy sigzz sigyz sigxz sigxy pressure equiv_stress" << endl;
    }

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
        ParticleVariable<double> plas, col, volume;
        ParticleVariable<Matrix3> stress;
        da->query(pos,    "p.x",              matl, patch, t);
        da->query(pID,    "p.particleID",     matl, patch, t);
        da->query(col,    "p.color",          matl, patch, t);
        bool havePS = false;
        if(have_plastic_strain){
         havePS = da->query(plas,  "p.plasticStrain",  matl, patch, t);
        }
        if(have_volume){
         da->query(volume,"p.volume",         matl, patch, t);
        }
        da->query(stress, "p.stress",         matl, patch, t);
        ParticleSubset* pset = pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            Matrix3 sig = stress[*iter];
            double pressure = (-1.0/3.0)*sig.Trace();
            double eqStress = sqrt(0.5*((sig(0,0)-sig(1,1))*(sig(0,0)-sig(1,1))
                            +           (sig(1,1)-sig(2,2))*(sig(1,1)-sig(2,2))
                            +           (sig(2,2)-sig(0,0))*(sig(2,2)-sig(0,0))
                            +      6.0*(sig(0,1)*sig(0,1) + sig(1,2)*sig(1,2) +
                                        sig(2,0)*sig(2,0))));

            partfile << pos[*iter].x() << " " << pos[*iter].y() << " " 
                     << pos[*iter].z() << " " << pID[*iter] << " " 
                     << col[*iter]         << " " << sig(0,0) << " " 
                     << sig(1,1) << " " << sig(2,2) << " " 
                     << sig(1,2) << " " << sig(0,2) << " "
                     << sig(0,1) << " " << pressure << " " << eqStress << " ";
            if(have_plastic_strain){
             if(havePS){
              partfile <<  plas[*iter] << " ";
             }else{
              partfile <<  0.0 << " ";
             }
            }
            if(have_volume){
              partfile <<  volume[*iter];
            }
            partfile << endl;
          } // for
        }  //if
      }  // for patches
    }   // for levels
   }    // for matls
  }     // for timesteps
} // end geocosm()

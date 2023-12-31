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

#include <StandAlone/tools/puda/aveColorDisp.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              aveColorDisp   O P T I O N
//  At each timestep, computes the average displacement for each color

void
Uintah::aveColorDisp( DataArchive * da, CommandLineFlags & clf )
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
//  for( int i = 0; i < (int)index.size(); i++ ) {
//    cout << index[i] << ": " << times[i] << endl;
//  }
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);

//  int matl = clf.matl;

  for(int matl=0; matl < clf.matl; matl++){ 
    // Make file for the data
    ostringstream fnum;
    fnum << matl;
    string filename = "TimeColorDisp_"+fnum.str()+".dat";
    ofstream outfile(filename.c_str());
    outfile.precision(12);

    // Make file for gnuplot 
    string plotfilename = "PlotFile_"+fnum.str()+".gplt";
    ofstream plotfile(plotfilename.c_str());

    plotfile << "set terminal x11 1" << endl;
    plotfile << "set autoscale" << endl;
    plotfile << "set xtics" << endl;
    plotfile << "set ytics" << endl;
    plotfile << "set key top left" << endl;
    plotfile << "set xlabel \"Time (us)\"" << endl;
    plotfile << "set ylabel \"Displacement Magnitude (cm)\"" << endl;
    plotfile << "plot \\" << endl;

    for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
      double time = times[t];
      cout << "time = " << time << endl;
      GridP grid = da->queryGrid(t);
      set<double> colors;
      LevelP level = grid->getLevel(grid->numLevels()-1);
      // Loop over all of the particles to find out what colors exist
      for(Level::const_patch_iterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<Point> value_pos;
        ParticleVariable<Vector> value_disp;
        ParticleVariable<double> value_color;
        da->query(value_pos, "p.x",            matl, patch, t);
        da->query(value_disp,"p.displacement", matl, patch, t);
        da->query(value_color,"p.color",       matl, patch, t);

        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator piter = pset->begin();
          for(;piter != pset->end(); piter++){
             colors.insert(value_color[*piter]);
          } // for
        }  //if
      }  // for patches

      if(t==0){
        cout << "Number of colors for material " << matl 
             << " = " << colors.size() << endl;
      }
      vector<double> ColorsVec(colors.size());
      vector<double> ColorDisp(colors.size());
      vector<int> NumOfColor(colors.size());
      unsigned long int i=0;
      if(t==0){
        outfile << "# Time "; 
      }
      for (set<double>::iterator it1 = colors.begin();
                                      it1!= colors.end();  it1++){
//        cout << "color = " << *it1 << endl;
        ColorsVec[i]=*it1;
        if(t==0){
          outfile << *it1 << " "; 
        }
        ColorDisp[i]  = 0.0;
        NumOfColor[i] = 0;
        if(t==0){
          plotfile << "\"TimeColorDisp_" << matl << ".dat\" using 1:" 
                   << i+2 << " w l t \"" << ColorsVec[i] << "\"";
          if(i<colors.size()-1){
            plotfile << ",\\";
          }
          plotfile << endl;
        }
        i++;
      } // for set iterator
      if(t==0){
        outfile << endl; 
      }

      // Now loop over all of the particles again
      for(Level::const_patch_iterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
//        int matl = clf.matl;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<Point> value_pos;
        ParticleVariable<Vector> value_disp;
        ParticleVariable<double> value_color;
        da->query(value_pos,  "p.x",            matl, patch, t);
        da->query(value_disp, "p.displacement", matl, patch, t);
        da->query(value_color,"p.color",        matl, patch, t);

        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator piter = pset->begin();
          for(;piter != pset->end(); piter++){
             for(long unsigned int ic = 0; ic<colors.size(); ic++){
               if(value_color[*piter]==ColorsVec[ic]){
                 ColorDisp[ic]+=value_disp[*piter].length();
                 NumOfColor[ic]++;
                 break;
               } // if
             }  // for colors
          } // for particles
        }  //if
      }  // for patches

      outfile << time << " "; 
      for(long unsigned int ic = 0; ic<colors.size(); ic++){
        ColorDisp[ic]/=((double) NumOfColor[ic]);
        outfile << ColorDisp[ic] << " "; 
      }
      outfile << endl; 
    } // loop over time
    plotfile << "pause -1" << endl;
  } // loop over matls
} // end aveColorDisp()


/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <StandAlone/tools/puda/pressure.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              P R E S S U R E   O P T I O N
//   This function will output the maximum pressure achieved during the
//   simulation as well as the average pressure and themaximum pressure 
//   at each timestep.   Timestep pressures are output to a file called
//   "Pressure.dat".

void
Uintah::pressure( DataArchive * da, CommandLineFlags & clf )
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
  string filename("Pressure.dat");
  ofstream outfile(filename.c_str());
  outfile << "Time    " << "Average Pressure    " << "Maximum Pressure" << endl;


  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
    double time = times[t];
    cout << "time = " << time << endl;
    GridP grid = da->queryGrid(t);
    double sumpressure = 0.00000;
    double ncells = 0.00000;
    double avgPress = 0.00000;
      double pressure = -9999999.0;  // the max pressure during the timestep
      LevelP level = grid->getLevel(grid->numLevels()-1);
      cout << "Level: " << grid->numLevels() - 1 <<  endl;
      for(Level::const_patch_iterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        int matl = clf.matl; // material number

        CCVariable<double> press_CC;
        // get all the pressures from the patch
        da->query(press_CC, "press_CC",        matl, patch, t);
        
        CCVariable<double> vol_frac_CC;
        // get all the Volume Fractions from the patch
        da->query(vol_frac_CC, "vol_frac_CC",        0, patch, t);

        for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
           IntVector c = *iter;  // get the coordinates of the cell
           
	   if(vol_frac_CC[c] > 0.5){
              sumpressure += press_CC[c]; //Solve for total pressure 
              ncells += 1; //Solve for total number of cells
           } // end if volume fraction

	   if(press_CC[c] > pressure){
              pressure = press_CC[c];
	   }   

          
        } // for cells
      }  // for patches
   avgPress = sumpressure/ncells; // solve for average pressure over all cells

   cout << "Max pressure for timestep was:\t" << pressure << endl;
   cout << "Average pressure for timestep was:\t" << avgPress << "\n" << endl;

   outfile.precision(15);
   outfile << time << " " << avgPress << " " <<pressure << endl; 

  } // for time
} // end pressure()


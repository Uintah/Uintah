/*
 
 The MIT License
 
 Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
 University of Utah.
 
 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a 
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation 
 the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included 
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 DEALINGS IN THE SOFTWARE.
 
 */


#include <StandAlone/tools/puda/jacquie.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              JACQUIE   O P T I O N
//   This function will output the burn rate achieved during the simulation  
//   simulation as well as the average pressure at each timestep.  Everything is
//   output to "AverageBurnRate.dat"
//   set matl to burning gas
//   Surface area and density is hard coded

void
Uintah::jacquie( DataArchive * da, CommandLineFlags & clf )
{
	vector<string> vars;
	vector<const Uintah::TypeDescription*> types;
	da->queryVariables(vars, types);
	ASSERTEQ(vars.size(), types.size());
	cout << "There are " << vars.size() << " variables:\n";
	for(int i=0;i<(int)vars.size();i++)
		cout << vars[i] << ": " << types[i]->getName() << endl;
	
	vector<int> index;
	vector<int> counts;
	vector<double> total_pr;
	
	
	vector<double> times;
	da->queryTimesteps(index, times);
	ASSERTEQ(index.size(), times.size());
	cout << "There are " << index.size() << " timesteps:\n";
	for( int i = 0; i < (int)index.size(); i++ ) {
		cout << index[i] << ": " << times[i] << endl;
	}
	
	findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
	
	ostringstream fnum;
	string filename("AverageBurnRate.dat");
	ofstream outfile(filename.c_str());
	double burnRate= 0;
	double avePress= 0;
	double total_press = 0;
	double total_burned_mass = 0.0;
	double SA = 0.001*0.001;
	int count;
	double minVol_frac = 0.9;  // the min volume fraction at any time
	
	
	for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
		double time = times[t];
		cout << "time = " << time << endl;
		GridP grid = da->queryGrid(t);
		count=0;
		total_press = 0;
		//double VolumeFraction = 0.001;  // the max pressure during the timestep
		LevelP level = grid->getLevel(grid->numLevels()-1);
		cout << "Level: " << grid->numLevels() - 1 <<  endl;
		for(Level::const_patchIterator iter = level->patchesBegin();
			iter != level->patchesEnd(); iter++){
			const Patch* patch = *iter;
			int matl = clf.matl_jim; // material number
			
			CCVariable<double> press_CC;
			CCVariable<double> vol_frac_CC;
			
			// get all the pressures from the patch
			da->query(press_CC, "press_CC",        0, patch, t);
			da->query(vol_frac_CC, "vol_frac_CC", matl, patch, t);
			for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
				IntVector c = *iter;  // get the coordinates of the cell
				
				if(vol_frac_CC[c] > minVol_frac){
					total_press += press_CC[c];
					count +=1;
					
				} //if
			} // cells
		} // patches
		counts.push_back(count);
		total_pr.push_back(total_press);
	}  // timesteps
	
	int p=0;
	
	for(unsigned long t=clf.time_step_lower+1;t<=clf.time_step_upper;t+=clf.time_step_inc){
		double time = times[t];
		double timeold = times[t-1];
		cout << "time = " << time << endl;
		GridP grid = da->queryGrid(t);
		total_burned_mass=0;
		LevelP level = grid->getLevel(grid->numLevels()-1);
		cout << "Level: " << grid->numLevels() - 1 <<  endl;
		for(Level::const_patchIterator iter = level->patchesBegin();
			iter != level->patchesEnd(); iter++){
			const Patch* patch = *iter;
			int matl = clf.matl_jim; // material number
			
			CCVariable<double> modelMass_src;
			// get all the modelMass from the patch
			da->query(modelMass_src, "modelMass_src",        matl, patch, t);
			
			for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
				IntVector c = *iter;  // get the coordinates of the cell
				total_burned_mass+= modelMass_src[c]; 
			} // for cells
		}  // for patch
		
		
		double density = 1832;
		burnRate = total_burned_mass/((time-timeold)*density*SA);
		avePress = total_pr[p]/counts[p];
		p++;
		
		outfile.precision(15);
		outfile << time << " " << avePress << " " << burnRate << " " <<total_press << " " << count << endl; 
    }//timestep
} // end jim2()

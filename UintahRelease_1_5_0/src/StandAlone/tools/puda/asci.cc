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



#include <StandAlone/tools/puda/asci.h>

#include <StandAlone/tools/puda/util.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/GridP.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

void
Uintah::asci( DataArchive *   da,
              const bool      tslow_set,
              const bool      tsup_set,
              unsigned long & time_step_lower,
              unsigned long & time_step_upper )
{
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;

  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  int freq = 1; int ts=1;
      
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  if (index.size() == 1) {
    cout << "There is only 1 timestep:\n"; 
  }
  else {
    cout << "There are " << index.size() << " timesteps:\n";
  }
      
  findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
      
  // Loop over time
  for( unsigned long t = time_step_lower; t <= time_step_upper; t++ ) {
    double time = times[t];
    int partnum = 1;
    int num_of_particles = 0;
    cout << "timestep " << ts << " inprogress... ";
	
    if( ( ts % freq) == 0 ) {
   		
      // dumps header and variable info to file
      //int variable_count =0;
      ostringstream fnum;
      string filename;
      int stepnum=ts/freq;
      fnum << setw(4) << setfill('0') << stepnum;
      string partroot("partout");
      filename = partroot+ fnum.str();
      ofstream partfile(filename.c_str());

      partfile << "TITLE = \"Time Step # " << time <<"\"," << endl;
                
      // Code to print out a list of Variables
      partfile << "VARIABLES = ";
	
      GridP grid = da->queryGrid(t);
      int l=0;
      LevelP level = grid->getLevel(l);
      Level::const_patchIterator iter = level->patchesBegin();
      const Patch* patch = *iter;
		
		
      // for loop over variables for name printing
      for(unsigned int v=0;v<vars.size();v++){
        std::string var = vars[v];
	       
        ConsecutiveRangeSet matls= da->queryMaterials(var, patch, t);
        // loop over materials
        for( ConsecutiveRangeSet::iterator matlIter = matls.begin(); matlIter != matls.end(); matlIter++ ) {
          int matl = *matlIter;
          const Uintah::TypeDescription* td = types[v];
          const Uintah::TypeDescription* subtype = td->getSubType();
          switch(td->getType()){
	        
            // The following only accesses particle data
          case Uintah::TypeDescription::ParticleVariable:
            switch(subtype->getType()){
            case Uintah::TypeDescription::double_type:
              {
                ParticleVariable<double> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
		      
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
			
                  if(matl == 0){
                    partfile << ", \"" << var << "\"";}
                  for(;iter != pset->end(); iter++){
                    num_of_particles++;
                  }
                }
                partnum=num_of_particles;
              }
            break;
            case Uintah::TypeDescription::float_type:
              {
                ParticleVariable<float> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
		      
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
			
                  if(matl == 0){
                    partfile << ", \"" << var << "\"";}
                  for(;iter != pset->end(); iter++){
                    num_of_particles++;
                  }
                }
                partnum=num_of_particles;
              }
            break;
            case Uintah::TypeDescription::Point:
              {
                ParticleVariable<Point> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
		      
                if(pset->numParticles() > 0 && (matl == 0)){
                  partfile << ", \"" << var << ".x\"" << ", \"" << var <<
                    ".y\"" << ", \"" <<var << ".z\"";
                }
              }
            break;
            case Uintah::TypeDescription::Vector:
              {
                ParticleVariable<Vector> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                //cout << td->getName() << " over " << pset->numParticles() << " particles\n";
                if(pset->numParticles() > 0 && (matl == 0)){
                  partfile << ", \"" << var << ".x\"" << ", \"" << var <<
                    ".y\"" << ", \"" << var << ".z\"";
                }
              }
            break;
            case Uintah::TypeDescription::Matrix3:
              {
                ParticleVariable<Matrix3> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                //cout << td->getName() << " over " << pset->numParticles() << " particles\n";
                if(pset->numParticles() > 0 && (matl == 0)){
                  partfile << ", \"" << var << ".1.1\"" << ", \"" << var << ".1.2\"" << ", \"" << var << ".1.3\""
                           << ", \"" << var << ".2.1\"" << ", \"" << var << ".2.2\"" << ", \"" << var << ".2.3\""
                           << ", \"" << var << ".3.1\"" << ", \"" << var << ".3.2\"" << ", \"" << var << ".3.3\"";
                }
              }
            break;
            default:
              cerr << "Particle Variable of unknown type: " << subtype->getName() << endl;
              break;
            }
            break;
          default:
            // Dd: Is this an error!?
            break;
          } // end switch( td->getType() )
		 
        } // end of for loop over materials

        // resets counter of number of particles, so it doesn't count for multiple
        // variables of the same type
        num_of_particles = 0;
	       
      } // end of for loop over variables
		
      partfile << endl << "ZONE I=" << partnum << ", F=BLOCK" << endl;	
		
      // Loop to print values for specific timestep
      // Because header has already been printed
		
      //variable initialization
      grid = da->queryGrid(t);
      level = grid->getLevel(l);
      iter = level->patchesBegin();
      patch = *iter;
	
      // loop over variables for printing values
      for(unsigned int v=0;v<vars.size();v++){
        std::string var = vars[v];
		
        ConsecutiveRangeSet matls=da->queryMaterials(var, patch, t);
        // loop over materials
        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
            matlIter != matls.end(); matlIter++){
          int matl = *matlIter;
          const Uintah::TypeDescription* td = types[v];
          const Uintah::TypeDescription* subtype = td->getSubType();
	        
          // the following only accesses particle data
          switch(td->getType()){
          case Uintah::TypeDescription::ParticleVariable:
            switch(subtype->getType()){
            case Uintah::TypeDescription::double_type:
              {
                ParticleVariable<double> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter] << " " << endl;
                  }
                  partfile << endl;
                }
              }
            break;
            case Uintah::TypeDescription::float_type:
              {
                ParticleVariable<float> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter] << " " << endl;
                  }
                  partfile << endl;
                }
              }
            break;
            case Uintah::TypeDescription::Point:
              {
                ParticleVariable<Point> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter].x() << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter].y() << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter].z() << " " << endl;
                  }  
                  partfile << endl;  
                }
              }
            break;
            case Uintah::TypeDescription::Vector:
              {
                ParticleVariable<Vector> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter].x() << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter].y() << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << value[*iter].z() << " " << endl;
                  }  
                  partfile << endl; 
                }
              }
            break;
            case Uintah::TypeDescription::Matrix3:
              {
                ParticleVariable<Matrix3> value;
                da->query(value, var, matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    partfile << (value[*iter])(0,0) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(0,1) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(0,2) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(1,0) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(1,1) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(1,2) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(2,0) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(2,1) << " " << endl;
                  }
                  partfile << endl;
                  iter = pset->begin();
                  for(;iter !=pset->end(); iter++){
                    partfile << (value[*iter])(2,2) << " " << endl;
                  }
                  partfile << endl;
                }
              }
            break;
            default:
              cerr << "Particle Variable of unknown type: " << subtype->getName() << endl;
              break;
            }
            break;
          default:
            // Dd: Is this an error?
            break;
          } // end switch( td->getType() )
        } // end of loop over materials 
      } // end of loop over variables for printing values
    } // end of if ts % freq	

    //increments to next timestep
    ts++;
    cout << " completed." << endl;
  } // end of loop over time

} // end asci()

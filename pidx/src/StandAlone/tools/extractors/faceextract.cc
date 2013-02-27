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

/*
 *  faceextract.cc: Create timeextract script for all cylinder faces
 *
 *  Written by:
 *   Stanislav Borodai
 *   Department of Chemical and Fuels Engineering 
 *   by stealing lineextract from:
 *   Jim Guilkey
 *   Department of Mechancial Engineering 
 *   University of Utah
 *   February 2007
 *
 */

#include <Core/DataArchive/DataArchive.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <cstdio>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

bool verbose = false;
bool d_printCell_coords = false;
bool d_donetheatflux = false;
bool d_dovelocity = false;
bool d_doincheatflux = false;
  
void
usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname << " [options] "
       << "-uda <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h,--help\n";
  cerr << "  -v,--variable <variable name> [defaults to cellType]\n";
  cerr << "  -tx,--timeextract <path to timeextract> [defaults to just timeextract]\n";
  cerr << "  -m,--material <material number> [defaults to 1]\n";
  cerr << "  -timestep,--timestep [int] (timestep used for face lookup int) [defaults to 1]\n";
  cerr << "  -container,--container [int] (container cell type int) [defaults to -3 (i.e. not present)]\n";
  cerr << "  -donetheatflux [do net heat flux extraction]\n";
  cerr << "  -dovelocity [do velocity extraction at face boundary (one off)]\n";
  cerr << "  -doincheatflux [do incident heat flux extraction]\n";
  cerr << "  -istart,--indexs <x> <y> <z> (cell index) [defaults to 0,0,0]\n";
  cerr << "  -iend,--indexe <x> <y> <z> (cell index) [defaults to Nx,Ny,Nz]\n";
  cerr << "  -l,--level [int] (level index to query range from) [defaults to 0]\n";
  cerr << "  -o,--out <outputfilename> [defaults to stdout]\n"; 
  cerr << "  -vv,--verbose (prints status of output)\n";
  //    cerr << "  -cellCoords (prints the cell centered coordinates on that level)\n";
  exit(1);
}

// arguments are the dataarchive, the successive arguments are the same as 
// the arguments to archive->query for data values.  Then comes a type 
// dexcription of the variable being queried, and last is an output stream.
//______________________________________________________________________
//
template<class T>
void printData(DataArchive* archive, string& variable_name,
               const Uintah::TypeDescription* variable_type,
               int material, int levelIndex,
               IntVector& var_start, IntVector& var_end,
               unsigned long timestep, int container, ostream& out,
               string& path_to_timeextract, string& input_uda_name) 

{
  // query time info from dataarchive
  vector<int> index;
  vector<double> times;

  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  if (verbose){
    cout << "There are " << index.size() << " timesteps\n";
  }
  
  //__________________________________
  // bullet proofing 
  if (timestep >= times.size() ) {
    cerr << "timestep must be between 0 and " << times.size()-1 << endl;
    exit(1);
  }
  
  // set defaults for output stream
  out.setf(ios::scientific,ios::floatfield);
  out.precision(16);
  
  bool cellNotFound = false;
  //__________________________________
  
  if (verbose)
    cout << "Outputting for time["<<timestep<<"] = " << times[timestep]<< endl;

  //__________________________________
  //  does the requested level exist
  bool levelExists = false;
  GridP grid = archive->queryGrid(timestep); 
  int numLevels = grid->numLevels();
   
  for (int L = 0;L < numLevels; L++) {
    const LevelP level = grid->getLevel(L);
    if (level->getIndex() == levelIndex){
      levelExists = true;
    }
  }
  if (!levelExists){
    cerr<< " Level " << levelIndex << " does not exist at this timestep " << timestep << endl;
  }
    
  if(levelExists){   // only extract data if the level exists
    const LevelP level = grid->getLevel(levelIndex);
    //__________________________________
    // User input starting and ending indicies    
        
    IntVector low, high;
    level->findCellIndexRange(low, high);
    if (var_start == IntVector(0,0,0)) {
      //  domain usually starts from 0,0,0, reset it anyway if not user
      //  specified
      var_start = low + IntVector(1,1,1);
    }
    if (var_end == IntVector(0,0,0)) {
      var_end = high - IntVector(2,2,2);
    }
    if (verbose) {
      cout <<" Search index from "<<var_start << " to " << var_end <<endl;
    }
          
    // find the corresponding patches
    Level::selectType patches;
    level->selectPatches(var_start, var_end + IntVector(1,1,1), patches);
    if( patches.size() == 0){
      cerr << " Could not find any patches on Level " << level->getIndex()
           << " that contain cells along line: " << var_start << " and " << var_end 
           << " Double check the starting and ending indices "<< endl;
      exit(1);
    }

    // query all the data up front
    vector<Variable*> vars(patches.size());
    for (int p = 0; p < patches.size(); p++) {
      switch (variable_type->getType()) {
      case Uintah::TypeDescription::CCVariable:
        vars[p] = scinew CCVariable<T>;
        archive->query( *(CCVariable<T>*)vars[p], variable_name, 
                        material, patches[p], timestep);
        break;
      case Uintah::TypeDescription::NCVariable:
        vars[p] = scinew NCVariable<T>;
        archive->query( *(NCVariable<T>*)vars[p], variable_name, 
                        material, patches[p], timestep);
        break;
      case Uintah::TypeDescription::SFCXVariable:
        vars[p] = scinew SFCXVariable<T>;
        archive->query( *(SFCXVariable<T>*)vars[p], variable_name, 
                        material, patches[p], timestep);
        break;
      case Uintah::TypeDescription::SFCYVariable:
        vars[p] = scinew SFCYVariable<T>;
        archive->query( *(SFCYVariable<T>*)vars[p], variable_name, 
                        material, patches[p], timestep);
        break;
      case Uintah::TypeDescription::SFCZVariable:
        vars[p] = scinew SFCZVariable<T>;
        archive->query( *(SFCZVariable<T>*)vars[p], variable_name, 
                        material, patches[p], timestep);
        break;
      default:
        cerr << "Unknown variable type: " << variable_type->getName() << endl;
      }
          
    }

    for (CellIterator ci(var_start, var_end+IntVector(1,1,1)); !ci.done(); ci++) {
      IntVector c = *ci;

      // find out which patch it's on (to keep the printing in sorted order.
      // alternatively, we could just iterate through the patches)
      int p = 0;
      for (; p < patches.size(); p++) {
        IntVector low = patches[p]->getExtraCellLowIndex();
        IntVector high = patches[p]->getExtraCellHighIndex();
        if (c.x() >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
            c.x() < high.x() && c.y() < high.y() && c.z() < high.z())
          break;
      }
      if (p == patches.size()) {
        cellNotFound = true;
        continue;
      }
      int p_xm = 0;
      for (; p_xm < patches.size(); p_xm++) {
        IntVector low = patches[p_xm]->getExtraCellLowIndex();
        IntVector high = patches[p_xm]->getExtraCellHighIndex();
        if (c.x()-1 >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
            c.x()-1 < high.x() && c.y() < high.y() && c.z() < high.z())
          break;
      }
      if (p_xm == patches.size()) {
        cellNotFound = true;
        continue;
      }
      int p_ym = 0;
      for (; p_ym < patches.size(); p_ym++) {
        IntVector low = patches[p_ym]->getExtraCellLowIndex();
        IntVector high = patches[p_ym]->getExtraCellHighIndex();
        if (c.x() >= low.x() && c.y()-1 >= low.y() && c.z() >= low.z() && 
            c.x() < high.x() && c.y()-1 < high.y() && c.z() < high.z())
          break;
      }
      if (p_ym == patches.size()) {
        cellNotFound = true;
        continue;
      }
      int p_zm = 0;
      for (; p_zm < patches.size(); p_zm++) {
        IntVector low = patches[p_zm]->getExtraCellLowIndex();
        IntVector high = patches[p_zm]->getExtraCellHighIndex();
        if (c.x() >= low.x() && c.y() >= low.y() && c.z()-1 >= low.z() && 
            c.x() < high.x() && c.y() < high.y() && c.z()-1 < high.z())
          break;
      }
      if (p_zm == patches.size()) {
        cellNotFound = true;
        continue;
      }
      int p_xp = 0;
      for (; p_xp < patches.size(); p_xp++) {
        IntVector low = patches[p_xp]->getExtraCellLowIndex();
        IntVector high = patches[p_xp]->getExtraCellHighIndex();
        if (c.x()+1 >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
            c.x()+1 < high.x() && c.y() < high.y() && c.z() < high.z())
          break;
      }
      if (p_xp == patches.size()) {
        cellNotFound = true;
        continue;
      }
      int p_yp = 0;
      for (; p_yp < patches.size(); p_yp++) {
        IntVector low = patches[p_yp]->getExtraCellLowIndex();
        IntVector high = patches[p_yp]->getExtraCellHighIndex();
        if (c.x() >= low.x() && c.y()+1 >= low.y() && c.z() >= low.z() && 
            c.x() < high.x() && c.y()+1 < high.y() && c.z() < high.z())
          break;
      }
      if (p_yp == patches.size()) {
        cellNotFound = true;
        continue;
      }
      int p_zp = 0;
      for (; p_zp < patches.size(); p_zp++) {
        IntVector low = patches[p_zp]->getExtraCellLowIndex();
        IntVector high = patches[p_zp]->getExtraCellHighIndex();
        if (c.x() >= low.x() && c.y() >= low.y() && c.z()+1 >= low.z() && 
            c.x() < high.x() && c.y() < high.y() && c.z()+1 < high.z())
          break;
      }
      if (p_zp == patches.size()) {
        cellNotFound = true;
        continue;
      }
          
      int val = -3;
      int val_xm = -3, val_xp = -3, val_ym = -3;
      int val_yp = -3, val_zm = -3, val_zp = -3;
      Vector dx = patches[p]->dCell();
      Vector shift(0,0,0);  // shift the cellPosition if it's a (X,Y,Z)FC variable
      switch (variable_type->getType()) {
      case Uintah::TypeDescription::CCVariable: 
        val = (*dynamic_cast<CCVariable<int>*>(vars[p]))[c]; 
        val_xm = (*dynamic_cast<CCVariable<int>*>(vars[p_xm]))[c-IntVector(1,0,0)]; 
        val_xp = (*dynamic_cast<CCVariable<int>*>(vars[p_xp]))[c+IntVector(1,0,0)]; 
        val_ym = (*dynamic_cast<CCVariable<int>*>(vars[p_ym]))[c-IntVector(0,1,0)]; 
        val_yp = (*dynamic_cast<CCVariable<int>*>(vars[p_yp]))[c+IntVector(0,1,0)]; 
        val_zm = (*dynamic_cast<CCVariable<int>*>(vars[p_zm]))[c-IntVector(0,0,1)]; 
        val_zp = (*dynamic_cast<CCVariable<int>*>(vars[p_zp]))[c+IntVector(0,0,1)]; 
        break;
      case Uintah::TypeDescription::NCVariable: 
        break;
      case Uintah::TypeDescription::SFCXVariable: 
        shift.x(-dx.x()/2.0); 
        break;
      case Uintah::TypeDescription::SFCYVariable: 
        shift.y(-dx.y()/2.0); 
        break;
      case Uintah::TypeDescription::SFCZVariable: 
        shift.z(-dx.z()/2.0); 
        break;
      default: break;
      }
          
      if(d_printCell_coords){
        Point point = level->getCellPosition(c);
        Vector here = point.asVector() + shift;
        out << here.x() << " "<< here.y() << " " << here.z() << " "<<val << endl;;
      }else if (d_donetheatflux){
        if ((val == container)&&(!(val_xm == container)))
          out << path_to_timeextract << " -v htfluxRadX -i "
              <<c.x() << " "<< c.y() << " " << c.z() <<" -o htfluxRadX_"
              <<c.x() << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_xp == container)))
          out << path_to_timeextract << " -v htfluxRadX -i "
              <<c.x()+1 << " "<< c.y() << " " << c.z() <<" -o htfluxRadX_"
              <<c.x()+1 << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_ym == container)))
          out << path_to_timeextract << " -v htfluxRadY -i "
              <<c.x() << " "<< c.y() << " " << c.z() <<" -o htfluxRadY_"
              <<c.x() << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_yp == container)))
          out << path_to_timeextract << " -v htfluxRadY -i "
              <<c.x() << " "<< c.y()+1 << " " << c.z() <<" -o htfluxRadY_"
              <<c.x() << "_"<< c.y()+1 << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_zm == container)))
          out << path_to_timeextract << " -v htfluxRadZ -i "
              <<c.x() << " "<< c.y() << " " << c.z() <<" -o htfluxRadZ_"
              <<c.x() << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_zp == container)))
          out << path_to_timeextract << " -v htfluxRadZ -i "
              <<c.x() << " "<< c.y() << " " << c.z()+1 <<" -o htfluxRadZ_"
              <<c.x() << "_"<< c.y() << "_" << c.z()+1<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
	  } else if (d_doincheatflux){
        if ((val == container)&&(!(val_xm == container)))
          out << path_to_timeextract << " -v radiationFluxEIN -i "
              <<c.x()-1 << " "<< c.y() << " " << c.z() <<" -o radiationFluxEIN_"
              <<c.x()-1 << "_"<< c.y() << "_" << c.z() <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_xp == container)))
          out << path_to_timeextract << " -v radiationFluxWIN -i "
              <<c.x()+1 << " "<< c.y() << " " << c.z() <<" -o radiationFluxWIN_"
              <<c.x()+1 << "_"<< c.y() << "_" << c.z() <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_ym == container)))
          out << path_to_timeextract << " -v radiationFluxNIN -i "
              <<c.x() << " "<< c.y()-1 << " " << c.z() <<" -o radiationFluxNIN_"
              <<c.x() << "_"<< c.y()-1 << "_" << c.z() <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_yp == container)))
          out << path_to_timeextract << " -v radiationFluxSIN -i "
              <<c.x() << " "<< c.y()+1 << " " << c.z() <<" -o radiationFluxSIN_"
              <<c.x() << "_"<< c.y()+1 << "_" << c.z() <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_zm == container)))
          out << path_to_timeextract << " -v radiationFluxTIN -i "
              <<c.x() << " "<< c.y() << " " << c.z()-1 <<" -o radiationFluxTIN_"
              <<c.x() << "_"<< c.y() << "_" << c.z()-1 <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        if ((val == container)&&(!(val_zp == container)))
          out << path_to_timeextract << " -v radiationFluxBIN -i "
              <<c.x() << " "<< c.y() << " " << c.z()+1 <<" -o radiationFluxBIN_"
              <<c.x() << "_"<< c.y() << "_" << c.z()+1 <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
      } else if (d_dovelocity){
        //Taking (delta x)/2 Cell Centered velocity components 
        if ((val == container)&&(!(val_xm == container))){
          out << path_to_timeextract << " -v newCCUVelocity -i "
              <<c.x()-1 << " "<< c.y() << " " << c.z() <<" -o newCCUVelocity_"
              <<c.x()-1 << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCVVelocity -i "
              <<c.x()-1 << " "<< c.y() << " " << c.z() <<" -o newCCVVelocity_"
              <<c.x()-1 << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCWVelocity -i "
              <<c.x()-1 << " "<< c.y() << " " << c.z() <<" -o newCCWVelocity_"
              <<c.x()-1 << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        }
        if ((val == container)&&(!(val_xp == container))){
          out << path_to_timeextract << " -v newCCUVelocity -i "
              <<c.x()+1 << " "<< c.y() << " " << c.z() <<" -o newCCUVelocity_"
              <<c.x()+1 << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCVVelocity -i "
              <<c.x()+1 << " "<< c.y() << " " << c.z() <<" -o newCCVVelocity_"
              <<c.x()+1 << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCWVelocity -i "
              <<c.x()+1 << " "<< c.y() << " " << c.z() <<" -o newCCWVelocity_"
              <<c.x()+1 << "_"<< c.y() << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        }
        if ((val == container)&&(!(val_ym == container))){
          out << path_to_timeextract << " -v newCCUVelocity -i "
              <<c.x() << " "<< c.y()-1 << " " << c.z() <<" -o newCCUVelocity_"
              <<c.x() << "_"<< c.y()-1 << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCVVelocity -i "
              <<c.x() << " "<< c.y()-1 << " " << c.z() <<" -o newCCVVelocity_"
              <<c.x() << "_"<< c.y()-1 << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCWVelocity -i "
              <<c.x() << " "<< c.y()-1 << " " << c.z() <<" -o newCCWVelocity_"
              <<c.x() << "_"<< c.y()-1 << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        }
        if ((val == container)&&(!(val_yp == container))){
          out << path_to_timeextract << " -v newCCUVelocity -i "
              <<c.x() << " "<< c.y()+1 << " " << c.z() <<" -o newCCUVelocity_"
              <<c.x() << "_"<< c.y()+1 << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCVVelocity -i "
              <<c.x() << " "<< c.y()+1 << " " << c.z() <<" -o newCCVVelocity_"
              <<c.x() << "_"<< c.y()+1 << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCWVelocity -i "
              <<c.x() << " "<< c.y()+1 << " " << c.z() <<" -o newCCWVelocity_"
              <<c.x() << "_"<< c.y()+1 << "_" << c.z()<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        }
        if ((val == container)&&(!(val_zm == container))){
          out << path_to_timeextract << " -v newCCUVelocity -i "
              <<c.x() << " "<< c.y() << " " << c.z()-1 <<" -o newCCUVelocity_"
              <<c.x() << "_"<< c.y() << "_" << c.z()-1<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCVVelocity -i "
              <<c.x() << " "<< c.y() << " " << c.z()-1 <<" -o newCCVVelocity_"
              <<c.x() << "_"<< c.y() << "_" << c.z()-1<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCWVelocity -i "
              <<c.x() << " "<< c.y() << " " << c.z()-1 <<" -o newCCWVelocity_"
              <<c.x() << "_"<< c.y() << "_" << c.z()-1<<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
        }
        if ((val == container)&&(!(val_zp == container))){
          out << path_to_timeextract << " -v newCCUVelocity -i "
              <<c.x() << " "<< c.y() << " " << c.z()+1 <<" -o newCCUVelocity_"
              <<c.x() << "_"<< c.y() << "_" << c.z()+1 <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCVVelocity -i "
              <<c.x() << " "<< c.y() << " " << c.z()+1 <<" -o newCCVVelocity_"
              <<c.x() << "_"<< c.y() << "_" << c.z()+1 <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl;
          out << path_to_timeextract << " -v newCCWVelocity -i "
              <<c.x() << " "<< c.y() << " " << c.z()+1 <<" -o newCCWVelocity_"
              <<c.x() << "_"<< c.y() << "_" << c.z()+1 <<".dat " 
              <<" -m "<<material<<" -uda "<<input_uda_name<<endl; 
        }

      }
    }
    for (unsigned i = 0; i < vars.size(); i++)
      delete vars[i];

  } // if level exists
    
} 


int main(int argc, char** argv)
{

  //__________________________________
  //  Default Values
  unsigned long timestep = 1;
  string input_uda_name;  
  string output_file_name("-");
  string path_to_timeextract = "timeextract";
  IntVector var_start(0,0,0);
  IntVector var_end(0,0,0);
  int levelIndex = 0;
  string variable_name = "cellType";
  int material = 1;
  int container = -3;
  
  //__________________________________
  // Parse arguments

  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if(s == "-tx" || s == "--timeextract") {
      path_to_timeextract = string(argv[++i]);
    } else if (s == "-m" || s == "--material") {
      material = atoi(argv[++i]);
    } else if (s == "-vv" || s == "--verbose") {
      verbose = true;
    } else if (s == "-timestep" || s == "--timestep") {
      int val = strtoul(argv[++i],(char**)NULL,10);
      timestep = val;
    } else if (s == "-container" || s == "--container") {
      int val = strtoul(argv[++i],(char**)NULL,10);
      container = val;
    } else if (s == "-istart" || s == "--indexs") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_start = IntVector(x,y,z);
    } else if (s == "-iend" || s == "--indexe") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_end = IntVector(x,y,z);
    } else if (s == "-l" || s == "--level") {
      levelIndex = atoi(argv[++i]);
    } else if( (s == "-h") || (s == "--help") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
    } else if (s == "-donetheatflux" ) {
      d_donetheatflux = true;
    } else if (s == "-doincheatflux") {
	  d_doincheatflux = true;		
	} else if ( s== "-dovelocity" ) {
      d_dovelocity = true;
    }else {
      usage(s, argv[0]);
    }
  }
  
  if(input_uda_name == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  // ---------------------------
  // Bullet proofing
  if (d_donetheatflux && d_dovelocity){
    cout << "Error!  You can only specify heat flux or velocity, not both\n";
    cout << "Aborting.\n";
    exit(-1);
  } else if (d_donetheatflux && d_doincheatflux) {
	cerr << "Error!  You can only specify incident or net heat flux (see help)\n";
	cerr << "Aborting.\n";
	exit(-1);
  }  else if (d_donetheatflux) {
    cout << "face extract for net heat flux \n";
  }  else if (d_dovelocity) {
    cout << "face extract for velocity \n";
  }  else if (d_doincheatflux) {
	cout << "face extract for incident heat flux \n";	  
  } else if (!(d_donetheatflux) && !(d_doincheatflux) && !(d_dovelocity)) {
    cerr << "You must specify -donetheatflux or -doincheatflux or -dovelocity (see help)" << endl;
	cerr << "Aborting!\n";
	exit(-1);
  }

  try {
    DataArchive* archive = scinew DataArchive(input_uda_name);
    
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    if (verbose) cout << "There are " << vars.size() << " variables:\n";
    bool var_found = false;
    unsigned int var_index = 0;
    for (;var_index < vars.size(); var_index++) {
      if (variable_name == vars[var_index]) {
        var_found = true;
        break;
      }
    }
    //__________________________________
    // bulletproofing
    if (!var_found) {
      cerr << "Variable \"" << variable_name << "\" was not found.\n";
      cerr << "If a variable name was not specified try -var [name].\n";
      cerr << "Possible variable names are:\n";
      var_index = 0;
      for (;var_index < vars.size(); var_index++) {
        cout << "vars[" << var_index << "] = " << vars[var_index] << endl;
      }
      cerr << "Aborting!!\n";
      exit(-1);
    }

    if (verbose) {
      cout << vars[var_index] << ": " << types[var_index]->getName() 
           << " being extracted for material "<<material << endl;
    }
    //__________________________________
    // get type and subtype of data
    const Uintah::TypeDescription* td = types[var_index];
    const Uintah::TypeDescription* subtype = td->getSubType();
     
    //__________________________________
    // Open output file, call printData with it's ofstream
    // if no output file, call with cout
    ostream *output_stream = &cout;
    if (output_file_name != "-") {
      if (verbose) cout << "Opening \""<<output_file_name<<"\" for writing.\n";
      ofstream *output = new ofstream();
      output->open(output_file_name.c_str());
      
      if (!(*output)) {   // bullet proofing
        cerr << "Could not open "<<output_file_name<<" for writing.\n";
        exit(1);
      }
      output_stream = output;
    } else {
      //output_stream = cout;
    }
    
    
    //__________________________________
    //  print data
    switch (subtype->getType()) {
    case Uintah::TypeDescription::double_type:
      break;
    case Uintah::TypeDescription::float_type:
      break;
    case Uintah::TypeDescription::int_type:
      printData<int>(archive, variable_name, td, material,
                     levelIndex, var_start, var_end,
                     timestep, container, *output_stream,
                     path_to_timeextract, input_uda_name);
      break;
    case Uintah::TypeDescription::Vector:
      break;
    case Uintah::TypeDescription::Other:
      if (subtype->getName() == "Stencil7") {
      }
      // don't break on else - flow to the error statement
    case Uintah::TypeDescription::Matrix3:
    case Uintah::TypeDescription::bool_type:
    case Uintah::TypeDescription::short_int_type:
    case Uintah::TypeDescription::long_type:
    case Uintah::TypeDescription::long64_type:
      cerr << "Subtype is not implemented\n";
      exit(1);
      break;
    default:
      cerr << "Unknown subtype\n";
      exit(1);
    }

    // Delete the output file if it was created.
    if (output_file_name != "-") {
      delete((ofstream*)output_stream);
    }

  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    exit(1);
  } catch(...){
    cerr << "Caught unknown exception\n";
    exit(1);
  }
}

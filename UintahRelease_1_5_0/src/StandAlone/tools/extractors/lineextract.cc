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
 *  lineextract.cc: Print out a uintah data archive
 *
 *  Written by:
 *   Jim Guilkey
 *   Department of Mechancial Engineering 
 *   by stealing timeextract from:
 *   James L. Bigler
 *   Bryan J. Worthen
 *   Department of Computer Science
 *   University of Utah
 *   June 2004
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
#include <Core/Grid/Variables/ParticleVariable.h>
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
bool quiet = false;
bool d_printCell_coords = false;
  
void
usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
        cerr << "Error parsing argument: " << badarg << endl;
    cerr << "Usage: " << progname << " [options] "
         << "-uda <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h,        --help\n";
    cerr << "  -v,        --variable:      <variable name>\n";
    cerr << "  -m,        --material:      <material number> [defaults to 0]\n";
    cerr << "  -tlow,     --timesteplow:   [int] (sets start output timestep to int) [defaults to 0]\n";
    cerr << "  -thigh,    --timestephigh:  [int] (sets end output timestep to int) [defaults to last timestep]\n";
    cerr << "  -timestep, --timestep:      [int] (only outputs from timestep int)  [defaults to 0]\n";
    cerr << "  -istart,   --indexs:        <i> <j> <k> [ints] starting point cell index  [defaults to 0 0 0]\n";
    cerr << "  -iend,     --indexe:        <i> <j> <k> [ints] end-point cell index [defaults to 0 0 0]\n";
    cerr << "  -startPt                    <x> <y> <z> [doubles] starting point of line in physical coordinates\n";
    cerr << "  -endPt                      <x> <y> <z> [doubles] end-point of line in physical coordinates\n";
	  cerr << "  -pr,       --precision:     [int] (specify precision of output data) [defaults to 16. maximum 32]\n";  
    cerr << "  -l,        --level:         [int] (level index to query range from) [defaults to 0]\n";
    cerr << "  -o,        --out:           <outputfilename> [defaults to stdout]\n"; 
    cerr << "  -vv,       --verbose:       (prints status of output)\n";
    cerr << "  -q,        --quiet:         (only print data values)\n";
    cerr << "  -cellCoords:                (prints the cell centered coordinates on that level)\n";
    cerr << "  --cellIndexFile:            <filename> (file that contains a list of cell indices)\n";
    cerr << "                                   [int 100, 43, 0]\n";
    cerr << "                                   [int 101, 43, 0]\n";
    cerr << "                                   [int 102, 44, 0]\n";
    cerr << "----------------------------------------------------------------------------------------\n";
    cerr << " For particle variables the average over all particles in a cell is returned.\n";
    exit(1);
}

// arguments are the dataarchive, the successive arguments are the same as 
// the arguments to archive->query for data values.  Then comes a type 
// dexcription of the variable being queried, and last is an output stream.


//______________________________________________________________________
//
template<class T>
void printData(DataArchive* archive, string& variable_name, const Uintah::TypeDescription* variable_type,
               int material, const bool use_cellIndex_file, int levelIndex,
               IntVector& var_start, IntVector& var_end, vector<IntVector> cells,
               unsigned long time_start, unsigned long time_end, unsigned long output_precision, ostream& out) 

{
  // query time info from dataarchive
  vector<int> index;
  vector<double> times;

  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  if (!quiet){
    cout << "There are " << index.size() << " timesteps\n";
  }
  
  // set default max time value
  if (time_end == (unsigned long)-1) {
    if (verbose) {
      cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
    }
    time_end = times.size() - 1;
  }      

  //__________________________________
  // bullet proofing 
  if (time_end >= times.size() || time_end < time_start) {
    cerr << "timestephigh("<<time_end<<") must be greater than " << time_start 
         << " and less than " << times.size()-1 << endl;
    exit(1);
  }
  if (time_start >= times.size() || time_end > times.size()) {
    cerr << "timestep must be between 0 and " << times.size()-1 << endl;
    exit(1);
  }
  
  //__________________________________
  // make sure the user knows it could be really slow if he
  // tries to output a big range of data...
  IntVector var_range = var_end - var_start;
  if (var_range.x() && var_range.y() && var_range.z()) {
    cerr << "PERFORMANCE WARNING: Outputting over 3 dimensions!\n";
  }
  else if ((var_range.x() && var_range.y()) ||
           (var_range.x() && var_range.z()) ||
           (var_range.y() && var_range.z())){
    cerr << "PERFORMANCE WARNING: Outputting over 2 dimensions\n";
  }

  // set defaults for output stream
  out.setf(ios::scientific,ios::floatfield);
  out.precision(output_precision);
  
  //__________________________________
  // loop over timesteps
  for (unsigned long time_step = time_start; time_step <= time_end; time_step++) {
  
    cerr << "%outputting for times["<<time_step<<"] = " << times[time_step]<< endl;

    //__________________________________
    //  does the requested level exist
    bool levelExists = false;
    GridP grid = archive->queryGrid(time_step); 
    int numLevels = grid->numLevels();
   
    for (int L = 0;L < numLevels; L++) {
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
    }
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }
    
    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      //__________________________________
      // User input starting and ending indicies    
      if(!use_cellIndex_file) {
          
        // find the corresponding patches
        Level::selectType patches;
        level->selectPatches(var_start, var_end + IntVector(1,1,1), patches,true);
        if( patches.size() == 0){
          cerr << " Could not find any patches on Level " << level->getIndex()
               << " that contain cells along line: " << var_start << " and " << var_end 
               << " Double check the starting and ending indices "<< endl;
          exit(1);
        }

        // query all the data up front
        vector<Variable*> vars(patches.size());
        for (int p = 0; p < patches.size(); p++) {
          if (patches[p]->isVirtual()) continue;
          switch (variable_type->getType()) {
          case Uintah::TypeDescription::CCVariable:
            vars[p] = scinew CCVariable<T>;
            archive->query( *(CCVariable<T>*)vars[p], variable_name, 
                            material, patches[p], time_step);
            break;
          case Uintah::TypeDescription::NCVariable:
            vars[p] = scinew NCVariable<T>;
            archive->query( *(NCVariable<T>*)vars[p], variable_name, 
                            material, patches[p], time_step);
            break;
          case Uintah::TypeDescription::SFCXVariable:
            vars[p] = scinew SFCXVariable<T>;
            archive->query( *(SFCXVariable<T>*)vars[p], variable_name, 
                            material, patches[p], time_step);
            break;
          case Uintah::TypeDescription::SFCYVariable:
            vars[p] = scinew SFCYVariable<T>;
            archive->query( *(SFCYVariable<T>*)vars[p], variable_name, 
                            material, patches[p], time_step);
            break;
          case Uintah::TypeDescription::SFCZVariable:
            vars[p] = scinew SFCZVariable<T>;
            archive->query( *(SFCZVariable<T>*)vars[p], variable_name, 
                            material, patches[p], time_step);
            break;
          default:
            cerr << "Unknown variable type: " << variable_type->getName() << endl;
          }
          
        }


        for (CellIterator ci(var_start, var_end + IntVector(1,1,1)); !ci.done(); ci++) {
          IntVector c = *ci;

          // find out which patch the variable is on
          int p = 0;
          for (; p < patches.size(); p++) {
            const Patch* patch = patches[p];
            
            if(patch->isVirtual()){
              continue;
            }
            
            T val = T();
            Vector dx = patch->dCell();
            Vector shift(0,0,0);  // shift the cellPosition if it's a (X,Y,Z)FC variable
            bool foundCell = false;
            
            switch (variable_type->getType()) {
            case Uintah::TypeDescription::CCVariable: 
              if(patch->containsCell(c)){
                val = (*dynamic_cast<CCVariable<T>*>(vars[p]))[c];
                foundCell = true; 
              }
            break;
            case Uintah::TypeDescription::NCVariable: 
              if(patch->containsNode(c)){
                val = (*dynamic_cast<NCVariable<T>*>(vars[p]))[c];
                foundCell = true; 
              }
            break;
            case Uintah::TypeDescription::SFCXVariable: 
              if(patch->containsSFCX(c)){
                val = (*dynamic_cast<SFCXVariable<T>*>(vars[p]))[c];
                shift.x(-dx.x()/2.0);
                foundCell = true;
              } 
            break;
            case Uintah::TypeDescription::SFCYVariable:
              if(patch->containsSFCY(c)){ 
                val = (*dynamic_cast<SFCYVariable<T>*>(vars[p]))[c];
                shift.y(-dx.y()/2.0); 
                foundCell = true;
              }
            break;
            case Uintah::TypeDescription::SFCZVariable: 
              if(patch->containsSFCY(c)){
                val = (*dynamic_cast<SFCZVariable<T>*>(vars[p]))[c];
                shift.z(-dx.z()/2.0); 
                foundCell = true;
              }
            break;
            default: break;
            }
            
            if(foundCell){
             if(d_printCell_coords){
                Point point = level->getCellPosition(c);
                Vector here = point.asVector() + shift;
                out << here.x() << " "<< here.y() << " " << here.z() << " "<<val << endl;;
              }else{
                out << c.x() << " "<< c.y() << " " << c.z() << " "<< val << endl;;
              }
            }
          } // patch loop
        }  // cell iterator
        
        
        for (unsigned i = 0; i < vars.size(); i++)
          delete vars[i];
      }

      //__________________________________
      // If the cell indicies were read from a file. 
      if(use_cellIndex_file) {
        for (int i = 0; i<(int) cells.size(); i++) {
          IntVector c = cells[i];
          vector<T> values;
          try {
            archive->query(values, variable_name, material, c, 
                            times[time_step], times[time_step], levelIndex);
          } catch (const VariableNotFoundInGrid& exception) {
            cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
            exit(1);
          }
          if(d_printCell_coords){
            Point p = level->getCellPosition(c);
            out << p.x() << " "<< p.y() << " " << p.z() << " "<< values[0] << endl;
          }else{
            out << c.x() << " "<< c.y() << " " << c.z() << " "<< values[0] << endl;
          }
        }
      }
      out << endl;
    } // if level exists
    
  } // timestep loop
}
//______________________________________________________________________
//  compute the average of all particles.
 
template<class T>
void compute_ave(ParticleVariable<T>& var,
                 CCVariable<T>& ave,
                 ParticleVariable<Point>& pos,
                 const Patch* patch) 
{
  IntVector lo = patch->getExtraCellLowIndex();
  IntVector hi = patch->getExtraCellHighIndex();
  ave.allocate(lo,hi);
  T zero(0);
  ave.initialize(zero);
  
  CCVariable<double> count;
  count.allocate(lo,hi);
  count.initialize(0.0);
  
  ParticleSubset* pset = var.getParticleSubset();
  if(pset->numParticles() > 0){
    ParticleSubset::iterator iter = pset->begin();
    for( ;iter != pset->end(); iter++ ){
      IntVector c;
      patch->findCell(pos[*iter], c);
      ave[c]    = ave[c] + var[*iter];
      count[c] += 1;      
    }
    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      ave[c] = ave[c]/(count[c] + 1e-100);
    }
  }
}
//______________________________________________________________________
// Used for Particle Variables
template<class T>
void printData_PV(DataArchive* archive, string& variable_name, const Uintah::TypeDescription* variable_type,
               int material, const bool use_cellIndex_file, int levelIndex,
               IntVector& var_start, IntVector& var_end, vector<IntVector> cells,
               unsigned long time_start, unsigned long time_end, unsigned long output_precision, ostream& out) 

{
  // query time info from dataarchive
  vector<int> index;
  vector<double> times;

  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  if (!quiet){
    cout << "There are " << index.size() << " timesteps\n";
  }
  
  // set default max time value
  if (time_end == (unsigned long)-1) {
    if (verbose) {
      cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
    }
    time_end = times.size() - 1;
  }      

  //__________________________________
  // bullet proofing 
  if (time_end >= times.size() || time_end < time_start) {
    cerr << "timestephigh("<<time_end<<") must be greater than " << time_start 
         << " and less than " << times.size()-1 << endl;
    exit(1);
  }
  if (time_start >= times.size() || time_end > times.size()) {
    cerr << "timestep must be between 0 and " << times.size()-1 << endl;
    exit(1);
  }
  
  //__________________________________IntVector c = cells[i];
  // make sure the user knows it could be really slow if he
  // tries to output a big range of data...
  IntVector var_range = var_end - var_start;
  if (var_range.x() && var_range.y() && var_range.z()) {
    cerr << "PERFORMANCE WARNING: Outputting over 3 dimensions!\n";
  }
  else if ((var_range.x() && var_range.y()) ||
           (var_range.x() && var_range.z()) ||
           (var_range.y() && var_range.z())){
    cerr << "PERFORMANCE WARNING: Outputting over 2 dimensions\n";
  }

  // set defaults for output stream
  out.setf(ios::scientific,ios::floatfield);
  out.precision(output_precision);
  
  //__________________________________
  // loop over timesteps
  for (unsigned long time_step = time_start; time_step <= time_end; time_step++) {
  
    cerr << "%outputting for times["<<time_step<<"] = " << times[time_step]<< endl;

    //__________________________________
    //  does the requested level exist
    bool levelExists = false;
    GridP grid = archive->queryGrid(time_step); 
    int numLevels = grid->numLevels();
   
    for (int L = 0;L < numLevels; L++) {
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
    }
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }
    
    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      
      // find the corresponding patches
      Level::selectType patches;
      level->selectPatches(var_start, var_end + IntVector(1,1,1), patches,true);
      if( patches.size() == 0){
        cerr << " Could not find any patches on Level " << level->getIndex()
             << " that contain cells along line: " << var_start << " and " << var_end 
             << " Double check the starting and ending indices "<< endl;
        exit(1);
      }

      // query all the data and compute the average up front
      vector<Variable*> vars(patches.size());
      vector<Variable*> ave(patches.size());
      for (int p = 0; p < patches.size(); p++) {
        vars[p] = scinew ParticleVariable<T>;
        ave[p]  = scinew CCVariable<T>;

        archive->query( *(ParticleVariable<T>*)vars[p], variable_name, 
                        material, patches[p], time_step);
        Variable* pos;
        pos = scinew ParticleVariable<Point>;    

        archive->query( *(ParticleVariable<Point>*)pos, "p.x", material, patches[p], time_step);

        compute_ave<T>(*(ParticleVariable<T>*)vars[p],
                       *(CCVariable<T>*)ave[p],
                       *(ParticleVariable<Point>*)pos,
                       patches[p]);
      }
      
      //__________________________________
      // User input starting and ending indicies    
      if(!use_cellIndex_file) {

        for (CellIterator ci(var_start, var_end + IntVector(1,1,1)); !ci.done(); ci++) {
          IntVector c = *ci;

          // find out which patch it's on (to keep the printing in sorted order.
          // alternatively, we could just iterate through the patches)
          int p = 0;
          for (; p < patches.size(); p++) {
            IntVector low  = patches[p]->getExtraCellLowIndex();
            IntVector high = patches[p]->getExtraCellHighIndex();
            if (c.x() >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
                c.x() < high.x() && c.y() < high.y() && c.z() < high.z())
              break;
          }
          if (p == patches.size()) {
            continue;
          }
          
          T val;
          val = (*dynamic_cast<CCVariable<T>*>(ave[p]))[c];
          
         if(d_printCell_coords){
            Point point = level->getCellPosition(c);
            out << point.x() << " "<< point.y() << " " << point.z() << " "<<val << endl;;
          }else{
            out << c.x() << " "<< c.y() << " " << c.z() << " "<< val << endl;;
          }
        }
        for (unsigned i = 0; i < vars.size(); i++)
          delete vars[i];
      }

      //__________________________________
      // If the cell indicies were read from a file. 
      if(use_cellIndex_file) {
        for (int i = 0; i<(int) cells.size(); i++) {
          IntVector c = cells[i];
          int p = 0;
          
          for (; p < patches.size(); p++) {
            IntVector low  = patches[p]->getExtraCellLowIndex();
            IntVector high = patches[p]->getExtraCellHighIndex();
            if (c.x() >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
                c.x() < high.x() && c.y() < high.y() && c.z() < high.z())
              break;
          }
          if (p == patches.size()) {
            continue;
          }
          
          T val;
          val = (*dynamic_cast<CCVariable<T>*>(ave[p]))[c];
          
          if(d_printCell_coords){
            Point point = level->getCellPosition(c);
            out << point.x() << " "<< point.y() << " " << point.z() << " "<< val << endl;
          }else{
            out << c.x() << " "<< c.y() << " " << c.z() << " "<< val << endl;
          }
        }
      } // if cell index file
      out << endl;
    } // if level exists
    
  } // timestep loop
}

/*_______________________________________________________________________
 Function:  readCellIndicies--
 Purpose: reads in a list of cell indicies
_______________________________________________________________________ */
void readCellIndicies(const string& filename, vector<IntVector>& cells)
{ 
  // open the file
  ifstream fp(filename.c_str());
  if (!fp){
    cerr << "Couldn't open the file that contains the cell indicies " << filename<< endl;
  }
  char c;
  int i,j,k;
  string text, comma;  
  
  while (fp >> c) {
    fp >> text>>i >> comma >> j >> comma >> k;
    IntVector indx(i,j,k);
    cells.push_back(indx);
    fp.get(c);
  }
  // We should do some bullet proofing here
  //for (int i = 0; i<(int) cells.size(); i++) {
  //  cout << cells[i] << endl;
  //}
}

//______________________________________________________________________
//    Notes:
// Now the material index is kind of a hard thing.  There is no way
// to reliably determine a default material.  Materials are defined
// on the patch for each varialbe, so this subset of materials could
// change over patches.  We can guess, that there will be a material
// 0.  This shouldn't cause the program to crash.  It will spit out
// an exception and exit gracefully.


int main(int argc, char** argv)
{

  //__________________________________
  //  Default Values
  bool use_cellIndex_file = false;
  bool findCellIndices = false;

  unsigned long time_start = 0;
  unsigned long time_end = (unsigned long)-1;
  unsigned long output_precision = 16;
  string input_uda_name;  
  string input_file_cellIndices;
  string output_file_name("-");
  IntVector var_start(0,0,0);
  IntVector var_end(0,0,0);
  Point     start_pt(-9,-9,-9);
  Point     end_pt(-9,-9,-9);
  int levelIndex = 0;
  vector<IntVector> cells;
  string variable_name;

  int material = 0;
  
  //__________________________________
  // Parse arguments

  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if (s == "-m" || s == "--material") {
      material = atoi(argv[++i]);
    } else if (s == "-vv" || s == "--verbose") {
      verbose = true;
    } else if (s == "-q" || s == "--quiet") {
      quiet = true;
    } else if (s == "-tlow" || s == "--timesteplow") {
      time_start = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-thigh" || s == "--timestephigh") {
      time_end = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-pr" || s == "--precision") {
      output_precision = strtoul(argv[++i],(char**)NULL,10);
      if (output_precision > 32) {
        std::cout << "Output precision cannot be larger than 32. Setting precision to 32 \n";        
        output_precision = 32;
      }
      if (output_precision < 1 ) {
        std::cout << "Output precision cannot be less than 1. Setting precision to 16 \n";
        output_precision = 16;
      }
    } else if (s == "-timestep" || s == "--timestep") {
      int val = strtoul(argv[++i],(char**)NULL,10);
      time_start = val;
      time_end = val;
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
    } else if (s == "-startPt" ) {
      double x = atof(argv[++i]);
      double y = atof(argv[++i]);
      double z = atof(argv[++i]);
      start_pt = Point(x,y,z);
    } else if (s == "-endPt" ) {
      double x = atof(argv[++i]);
      double y = atof(argv[++i]);
      double z = atof(argv[++i]);
      end_pt = Point(x,y,z);
      findCellIndices = true;
    } else if (s == "-l" || s == "--level") {
      levelIndex = atoi(argv[++i]);
    } else if( (s == "-h") || (s == "--help") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
    } else if (s == "--cellIndexFile") {
      use_cellIndex_file = true;
      input_file_cellIndices = string(argv[++i]);
    } else if (s == "--cellCoords" || s == "-cellCoords" ) {
      d_printCell_coords = true;
    }else {
      usage(s, argv[0]);
    }
  }
  
  if(input_uda_name == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
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
    //  find the cell index
    if(findCellIndices){
      vector<int> index;
      vector<double> times;
      archive->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());

      GridP grid = archive->queryGrid(time_start);
      const LevelP level = grid->getLevel(levelIndex);  
      if (level){                                       
        var_start=level->getCellIndex(start_pt);
        var_end  =level->getCellIndex(end_pt);                    
      }
    }                                  

    if (!quiet) {
      cout << vars[var_index] << ": " << types[var_index]->getName() 
           << " being extracted for material "<<material
           <<" at index "<<var_start << " to " << var_end <<endl;
    }    
    //__________________________________    
    // read in cell indices from a file
    if ( use_cellIndex_file) {
      readCellIndicies(input_file_cellIndices, cells);
    }
    
    //__________________________________
    //  print data
    //  N C / C C   V A R I A B L E S  
    if(td->getType() != Uintah::TypeDescription::ParticleVariable){
      switch (subtype->getType()) {
      case Uintah::TypeDescription::double_type:
        printData<double>(archive, variable_name, td, material, use_cellIndex_file,
                          levelIndex, var_start, var_end, cells,
                          time_start, time_end, output_precision, *output_stream);
        break;
      case Uintah::TypeDescription::float_type:
        printData<float>(archive, variable_name, td, material, use_cellIndex_file,
                          levelIndex, var_start, var_end, cells,
                          time_start, time_end, output_precision, *output_stream);
        break;
      case Uintah::TypeDescription::int_type:
        printData<int>(archive, variable_name, td, material, use_cellIndex_file,
                       levelIndex, var_start, var_end, cells,
                       time_start, time_end, output_precision, *output_stream);
        break;
      case Uintah::TypeDescription::Vector:
        printData<Vector>(archive, variable_name, td, material, use_cellIndex_file,
                          levelIndex, var_start, var_end, cells,
                          time_start, time_end, output_precision, *output_stream);    
        break;
      case Uintah::TypeDescription::Matrix3:
       printData<Matrix3>(archive, variable_name, td, material, use_cellIndex_file,
                         levelIndex, var_start, var_end, cells,
                         time_start, time_end, output_precision, *output_stream);    
        break;
      case Uintah::TypeDescription::Other:
        if (subtype->getName() == "Stencil7") {
          printData<Stencil7>(archive, variable_name, td, material, use_cellIndex_file,
                            levelIndex, var_start, var_end, cells,
                            time_start, time_end, output_precision, *output_stream);    
          break;
        }
        // don't break on else - flow to the error statement
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
    }
    //__________________________________
    //  P A R T I C L E   V A R I A B L E  
    if(td->getType() == Uintah::TypeDescription::ParticleVariable){
      switch (subtype->getType()) {
      case Uintah::TypeDescription::double_type:
        printData_PV<double>(archive, variable_name, td, material, use_cellIndex_file,
                          levelIndex, var_start, var_end, cells,
                          time_start, time_end, output_precision, *output_stream);
        break;
      case Uintah::TypeDescription::float_type:
        printData_PV<float>(archive, variable_name, td, material, use_cellIndex_file,
                          levelIndex, var_start, var_end, cells,
                          time_start, time_end, output_precision, *output_stream);
        break;
      case Uintah::TypeDescription::int_type:
        printData_PV<int>(archive, variable_name, td, material, use_cellIndex_file,
                       levelIndex, var_start, var_end, cells,
                       time_start, time_end, output_precision, *output_stream);
        break;
      case Uintah::TypeDescription::Vector:
        printData_PV<Vector>(archive, variable_name, td, material, use_cellIndex_file,
                          levelIndex, var_start, var_end, cells,
                          time_start, time_end, output_precision, *output_stream);    
        break;
      case Uintah::TypeDescription::Other:
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

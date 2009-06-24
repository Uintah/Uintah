/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


/*
 *  compute_Lnorm_udas.cc: compare the results of 2 udas and compute an L norm
 *  The udas don't have to be the same size.
 *
 *  Written by:
 *   Todd Harman
 *   Department of Mechanical Engineering
 *   University of Utah
 *   February 2009
 *
 *  Copyright (C) 2009 U of U
 */

#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/ProgressiveWarning.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <cmath>
using namespace SCIRun;
using namespace std;
using namespace Uintah;


void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "")
    cerr << "\nError parsing argument: " << badarg << '\n';
  cerr << "\nUsage: " << progname 
       << " [options] <archive file 1> <archive file 2>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h[elp]\n";
  cerr << "  -as_warnings             (treat tolerance errors as warnings and continue)\n";
  cerr << "  -skip_unknown_types      (skip variable comparisons of unknown types without error)\n";
  cerr << "  -ignoreVariable [string] (skip this variable)\n";
  cerr << "  -dont_sort               (don't sort the variable names before comparing them)";
  cerr << "\nNote: The absolute and relative tolerance tests must both fail\n"
       << "      for a comparison to fail.\n\n";
  Thread::exitAll(1);
}

// I don't want to have to pass these parameters all around, so
// I've made them file-scope.
string filebase1;
string filebase2;
bool strict_types = true;

enum Norm {
    L1, L2, LInfinity
};


void abort_uncomparable()
{
  cerr << "\nThe uda directories may not be compared.\n";
  Thread::exitAll(5);
}
#if 0
void initialize(double a, double value)
{
  a = value;
}
void initialize(Vector a, double value)
{
  a = Vector(value, value, value);
}

bool norm(double a, double b, double sum)
{
  return sum += Abs(a) - Abs(b);
}


bool norm(Vector a, Vector b, Vector sum)
{
  return norm(a.x(), b.x(), sum.x()) &&
         norm(a.y(), b.y(), sum.y()) &&
         norm(a.z(), b.z(), sum.z());
}
#endif
//__________________________________
//
GridIterator
getIterator( const Uintah::TypeDescription * td, const Patch * patch, bool use_extra_cells ) 
{
  switch( td->getType() ){
    case Uintah::TypeDescription::NCVariable :    return GridIterator( patch->getNodeIterator__New() );
    case Uintah::TypeDescription::CCVariable :    return GridIterator( patch->getCellIterator__New() );
    case Uintah::TypeDescription::SFCXVariable :  return GridIterator( patch->getSFCXIterator__New() );
    case Uintah::TypeDescription::SFCYVariable :  return GridIterator( patch->getSFCYIterator__New() );
    case Uintah::TypeDescription::SFCZVariable :  return GridIterator( patch->getSFCZIterator__New() );
    default:
      cout << "ERROR: Don't know how to handle type: " << td->getName() << "\n";
      exit( 1 );
  }
} // end getIterator()

//__________________________________
template <class Field, class subtype>
void compareFields(DataArchive* da1, 
                   DataArchive* da2, 
                   const string& var,
                   const int matl,
                   const Patch* patch,
                   const Array3<const Patch*>& cellToPatchMap,
                   int timestep)
{
  Field field2;
  Field field1;
  subtype sum(0);
  subtype max(0);
 
  const Uintah::TypeDescription * td1 = field1.getTypeDescription();
  GridIterator iter = getIterator( td1, patch, false);
  cout << " iter " << iter <<endl;
  
  ConsecutiveRangeSet matls1 = da1->queryMaterials(var, patch, timestep);
  
  da1->query(field1, var, matl, patch, timestep);

  map<const Patch*, Field*> patch2FieldMap;
  typename map<const Patch*, Field*>::iterator findIter;
#if 0
  for( ; !iter.done(); iter++ ) {
    IntVector c = *iter;
    const Patch* patch2 = cellToPatchMap[c];
    
    ConsecutiveRangeSet matls2 = da2->queryMaterials(var, patch2, timestep);
    ASSERT(matls1 == matls2);

    findIter = patch2FieldMap.find(patch2);
    
    // load in data
    if (findIter == patch2FieldMap.end()) {
      field2 = scinew Field();
      patch2FieldMap[patch2] = field2;
      da2->query(*field2, var, matl, patch2, timestep);
    }else {
      field2 = (*findIter).second;
    }

    sum += field1[c];
    max = Max(max(field1[c]));
    cout << " *iter " << c << endl;
    //norm(field[c], (*field2)[c])
  }
#endif
  cout << "sum " << sum << " max " << max << endl;

  typename map<const Patch*, Field*>::iterator itr = patch2FieldMap.begin();
  for ( ; itr != patch2FieldMap.end(); itr++) {
    delete (*itr).second;
  }


}


//__________________________________
// function loop
void BuildCellToPatchMap(LevelP level, 
                         const string& filebase,
                         Array3<const Patch*>& patchMap, 
                         double time, 
                         Patch::VariableBasis basis)
{
  const PatchSet* allPatches = level->allPatches();
  const PatchSubset* patches = allPatches->getUnion();
  if (patches->size() == 0){
    return;
  }

  IntVector bl=IntVector(0,0,0);
  IntVector low  = patches->get(0)->getExtraLowIndex(basis,bl);
  IntVector high = patches->get(0)->getExtraHighIndex(basis,bl);

  for (int i = 1; i < patches->size(); i++) {
    low  = Min(low,  patches->get(i)->getExtraLowIndex(basis,bl));
    high = Max(high, patches->get(i)->getExtraHighIndex(basis,bl));
  }
  
  patchMap.resize(low, high);
  patchMap.initialize(0);

  Level::const_patchIterator patch_iter;
  for(patch_iter = level->patchesBegin(); patch_iter != level->patchesEnd(); patch_iter++) {
    const Patch* patch = *patch_iter;
    
    ASSERT(Min(patch->getExtraLowIndex(basis,bl), low) == low);
    ASSERT(Max(patch->getExtraHighIndex(basis,bl), high) == high);
    
    patchMap.rewindow(patch->getExtraLowIndex(basis,bl),
                      patch->getExtraHighIndex(basis,bl));
    
    for (Array3<const Patch*>::iterator iter = patchMap.begin(); iter != patchMap.end(); iter++) {
      if (*iter != 0) {
        static ProgressiveWarning pw("Two patches on the same grid overlap", 10);
        if (pw.invoke())
          cerr << "Patches " << patch->getID() << " and " 
               << (*iter)->getID() << " overlap on the same file at time " << time
               << " in " << filebase << " at index " << iter.getIndex() << endl;
        //abort_uncomparable();
        
        // in some cases, we can have overlapping patches, where an extra cell/node 
        // overlaps an interior cell/node of another patch.  We prefer the interior
        // one.  (if there are two overlapping interior ones (nodes or face centers only),
        // they should have the same value.  However, since this patchMap is also used 
        // for cell centered variables give priority to the patch that has this index within 
        // its interior cell centered variables
        IntVector in_low  = patch->getLowIndex(basis);
        IntVector in_high = patch->getHighIndex(basis);
        IntVector pos = iter.getIndex();
        
        if (pos.x() >= in_low.x() && pos.y() >= in_low.y() && pos.z() >= in_low.z() &&
            pos.x() < in_high.x() && pos.y() < in_high.y() && pos.z() < in_high.z()) {
          *iter = patch;
        }
      }else{
        IntVector pos = iter.getIndex();
        *iter = patch;
      }
    }  //patchMap
  }  //level
  patchMap.rewindow(low, high);
}

//______________________________________________________________________
int
main(int argc, char** argv)
{
  Thread::setDefaultAbortMode("exit");
  string ignoreVar = "none";
  bool sortVariables = true;

  //__________________________________
  // Parse Args:
  for( int i = 1; i < argc; i++ ) {
    string s = argv[i];
    if(s == "-dont_sort") {
      sortVariables = false;
    }
    else if(s == "-skip_unknown_types") {
      strict_types = false;
    }
    else if(s == "-ignoreVariable") {
      if (++i == argc){
        usage("-ignoreVariable, no variable given", argv[0]);
      }else{
        ignoreVar = argv[i];
      }
    }
    else if(s[0] == '-' && s[1] == 'h' ) { // lazy check for -h[elp] option
      usage( "", argv[0] );
    }
    else {
      if (filebase1 != "") {
        if (filebase2 != "")
          usage(s, argv[0]);
        else
          filebase2 = argv[i];
      }
      else
        filebase1 = argv[i];
    }
  }
  //__________________________________
  if( filebase2 == "" ){
    cerr << "\nYou must specify two archive directories.\n";
    usage("", argv[0]);
  }

  DataArchive* da1 = scinew DataArchive(filebase1);
  DataArchive* da2 = scinew DataArchive(filebase2);

  vector<string> vars, vars2;
  vector<const Uintah::TypeDescription*> types, types2;
  vector< pair<string, const Uintah::TypeDescription*> > vartypes1,vartypes2;    
  da1->queryVariables(vars,  types);
  da2->queryVariables(vars2, types2);

  ASSERTEQ(vars.size(),  types.size());
  ASSERTEQ(vars2.size(), types2.size());

  if (vars.size() != vars2.size()) {
    cerr << filebase1 << " has " << vars.size() << " variables\n";
    cerr << filebase2 << " has " << vars2.size() << " variables\n";
    abort_uncomparable();
  }

  //__________________________________
  // bulletproofing    
  for (unsigned int i = 0; i < vars.size(); i++) {
    if (vars[i] != vars2[i]) {
      cerr << "Variable " << vars[i]  << " in " << filebase1 << " does not match\n";
      cerr << "variable " << vars2[i] << " in " << filebase2 << endl;
      abort_uncomparable();
    }

    if (types[i] != types2[i]) {
      cerr << "Variable " << vars[i] << " does not have the same type in both uda directories.\n";
      cerr << "In " << filebase1 << " its type is " << types[i]->getName() << endl;
      cerr << "In " << filebase2 << " its type is " << types2[i]->getName() << endl;
      abort_uncomparable();
    } 
  }

  //__________________________________
  // Look over timesteps
  vector<int> index, index2;
  vector<double> times, times2;

  da1->queryTimesteps(index,  times);
  da2->queryTimesteps(index2, times2);

  ASSERTEQ(index.size(),  times.size());
  ASSERTEQ(index2.size(), times2.size());

  for(unsigned long t = 0; t < times.size() && t < times2.size(); t++){

    double time1 = times[t];
    double time2 = times2[t];
    cerr << "time = " << time1 << "\n";
    GridP grid  = da1->queryGrid(t);
    GridP grid2 = da2->queryGrid(t);

    // bullet proofing
    if (grid->numLevels() != grid2->numLevels()) {
      cerr << "Grid at time " << time1 << " in " << filebase1 << " has " << grid->numLevels() << " levels.\n";
      cerr << "Grid at time " << time2 << " in " << filebase2 << " has " << grid2->numLevels() << " levels.\n";
      abort_uncomparable();
    }
    if (abs(times[t] - times2[t]) > 1e-5) {
      cerr << "Timestep at time " << times[t] << " in " << filebase1 << " does not match\n";
      cerr << "timestep at time " << times2[t] << " in " << filebase2 << " within the allowable tolerance.\n";
      abort_uncomparable();
    }


    //__________________________________
    //  Loop over variables
    for(int v=0;v<(int)vars.size();v++){
      std::string var = vars[v];
      const Uintah::TypeDescription* td = types[v];
      const Uintah::TypeDescription* subtype = td->getSubType();

      cerr << "\tVariable: " << var << ", type " << td->getName() << "\n";

      Patch::VariableBasis basis=Patch::translateTypeToBasis(td->getType(),false);

      for(int l=0;l<grid->numLevels();l++){
        LevelP level   = grid->getLevel(l);
        LevelP level2  = grid2->getLevel(l);

        //check patch coverage
        vector<Region> region1, region2, difference1, difference2;

        for(int i=0;i<level->numPatches();i++){
          const Patch* patch=level->getPatch(i);
          region1.push_back(Region(patch->getExtraCellLowIndex__New(),patch->getExtraCellHighIndex__New()));
        }

        for(int i=0;i<level2->numPatches();i++){
          const Patch* patch=level2->getPatch(i);
          region2.push_back(Region(patch->getExtraCellLowIndex__New(),patch->getExtraCellHighIndex__New()));
        }

        difference1 = Region::difference(region1,region2);
        difference2 = Region::difference(region1,region2);

        if(!difference1.empty() || !difference2.empty()){
          cerr << "Patches on level:" << l << " do not cover the same area\n";
          abort_uncomparable();
        }

        // map cells to patches in level and level2 respectively
        Array3<const Patch*> cellToPatchMap;
        Array3<const Patch*> cellToPatchMap2;

        BuildCellToPatchMap(level,  filebase1, cellToPatchMap,  time1, basis);
        BuildCellToPatchMap(level2, filebase2, cellToPatchMap2, time2, basis);

        // bulletproofing
        for ( Array3<const Patch*>::iterator nodePatchIter = cellToPatchMap.begin(); nodePatchIter != cellToPatchMap.end(); nodePatchIter++) {

          IntVector index = nodePatchIter.getIndex();

          if ((cellToPatchMap[index]  == 0 && cellToPatchMap2[index] != 0) ||
              (cellToPatchMap2[index] == 0 && cellToPatchMap[index] != 0)) {
            cerr << "Inconsistent patch coverage on level " << l << " at time " << time1 << endl;

            if (cellToPatchMap[index] != 0) {
              cerr << index << " is covered by " << filebase1 << " and not " << filebase2 << endl;
            } else {
              cerr << index << " is covered by " << filebase2 << " and not " << filebase1 << endl;
            }

            abort_uncomparable();
          }
        }

        //__________________________________
        //  compare the fields
        
        //ConsecutiveRangeSet matls = da1->queryMaterials(var, patch, timestep);
        //ConsecutiveRangeSet::iterator matlIter;
        //for ( matlIter = matls.begin() ;matlIter != matls.end(); matlIter++){
        //int matl = *matlIter;
          int matl = 0;
          Level::const_patchIterator iter;
          for(iter = level->patchesBegin();iter != level->patchesEnd(); iter++) {
            const Patch* patch = *iter;

            switch(td->getType()){
              case Uintah::TypeDescription::CCVariable:                                                   
                switch(subtype->getType()){                                                                 
                  case Uintah::TypeDescription::int_type:{ 
                    cout <<" int " << endl;                                                                   
                    compareFields<CCVariable<int>,int>(da1, da2, var, matl, patch, cellToPatchMap2, t);        
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    cout << "double" << endl;
                    compareFields<CCVariable<double>,double>(da1, da2, var, matl, patch, cellToPatchMap2, t); 
                    break;
                  }               
                  case Uintah::TypeDescription::Vector:{
                    cout <<" vector " << endl;
                    compareFields<CCVariable<Vector>,Vector>(da1, da2, var, matl, patch, cellToPatchMap2, t);  
                    break;
                  }
                }
                break; 
              default:
                cerr << "Variable of unknown type: " << td->getName() << endl;
              break;
            }
          }  // patches
        //}  // matls
      }  // levels
    }  // variables
  }

  if (times.size() != times2.size()) {
    cerr << endl;
    cerr << filebase1 << " has " << times.size() << " timesteps\n";
    cerr << filebase2 << " has " << times2.size() << " timesteps\n";
    abort_uncomparable();
  }
  delete da1;
  delete da2;
  return 0;
}

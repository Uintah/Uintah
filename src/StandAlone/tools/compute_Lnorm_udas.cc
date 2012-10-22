/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
 *  compute_Lnorm_udas.cc: compare the results of 2 udas and compute an L norm
 *  The udas don't have to be the same size.
 *
 *  Written by:
 *   Todd Harman
 *   Department of Mechanical Engineering
 *   University of Utah
 *   February 2009
 *
 */

#include <Core/DataArchive/DataArchive.h>
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
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Thread/Thread.h>
#include <Core/Util/ProgressiveWarning.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace Uintah;

void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != ""){
    cerr << "\nError parsing argument: " << badarg << "\n\n";
  }
  cout << " \n compute_Lnorm_uda:  Computes the L(1,2,inf) norm for each variable in two udas.  Each variable in uda1 \n"
       << " is examined at each level and timestep.  You can compare udas that have different computational domains\n"
       << " and different patch distributions.  The uda with the small compuational domain should always be specified first.\n \n"
       << " Output is sent to the terminal in addition to a directory named 'Lnorm'.  The structure of 'Lnorm' is:\n"
       << " Lnorm/ \n"
       << "   -- L-0 \n"
       << "    |-- delP_Dilatate_0 \n"
       << "    |-- mom_L_ME_CC_0 \n"
       << "    |-- press_CC_0 \n"
       << "    |-- press_equil_CC_0 \n"
       << "In each file is the physical time, L1, L2, Linf norms.\n";
       
  cout << "\nNote only CC, NC, SFCX, SFCY, SFCZ variables are supported.\n";      
  cout << "\nUsage: "
       << " [options] <uda1> <uda2>\n\n";
  cout << "Valid options are:\n";
  cout << "  -h[elp]\n";
  Thread::exitAll(1);
}
//__________________________________
void abort_uncomparable()
{
  cerr << "\nThe uda directories may not be compared.\n";
  Thread::exitAll(5);
}

//__________________________________
Vector Sqrt(const Vector a)
{
  return Vector(Sqrt(a.x()), Sqrt(a.y()), Sqrt(a.z()));
}

//__________________________________
template <class T>
class Norms{
  public:
    T d_L1;
    T d_L2;
    T d_Linf;  
    int d_n;
    
    void getNorms( T& L1, T& L2, T& Linf, int& n){
      L1 = d_L1;
      L2 = d_L2;
      Linf = d_Linf;
      n = d_n;
    };

    void setNorms (const T L1, const T L2, const T Linf, const int n)
    {
      d_n = n;
      d_L1 = L1;
      d_L2 = L2;
      d_Linf = Linf;
    }
    
    void printNorms() 
    {
      d_L1   = d_L1/((T)d_n);
      d_L2   = Sqrt( d_L2/((T)d_n) );
      cout << " \t norms: L1 " << d_L1 << " L2: " << d_L2 << " Linf: " << d_Linf<< " n_cells: " << d_n<<endl;
    }
    
    void outputNorms(const double& time,const string& filename) 
    {
      ofstream out(filename.c_str(), ios_base::app);
      
      out.setf(ios::scientific,ios::floatfield);
      out.precision(10);
      
      if(! out){
        cerr << " could not open output file: " << filename << endl;
        abort_uncomparable();
      }
      out << time << " " << d_L1 << " " << d_L2 << " " << d_Linf<<endl;
      out.close();
    }
    
    int get_n()
    {
      return d_n;
    }
};

//__________________________________
//
GridIterator
getIterator( const Uintah::TypeDescription * td, const Patch * patch, bool use_extra_cells ) 
{
  switch( td->getType() ){
    case Uintah::TypeDescription::NCVariable :    return GridIterator( patch->getNodeIterator() );
    case Uintah::TypeDescription::CCVariable :    return GridIterator( patch->getCellIterator() );
    case Uintah::TypeDescription::SFCXVariable :  return GridIterator( patch->getSFCXIterator() );
    case Uintah::TypeDescription::SFCYVariable :  return GridIterator( patch->getSFCYIterator() );
    case Uintah::TypeDescription::SFCZVariable :  return GridIterator( patch->getSFCZIterator() );
    default:
      cout << "ERROR: Don't know how to handle type: " << td->getName() << "\n";
      exit( 1 );
  }
} // end getIterator()

//__________________________________
template <class Field, class subtype>
void compareFields(Norms<subtype>* norms,
                   DataArchive* da1, 
                   DataArchive* da2, 
                   const string& var,
                   const int matl,
                   const Patch* patch,
                   const Array3<const Patch*>& cellToPatchMap,
                   int timestep)
{
  Field* field2;
  Field field1;
  
  subtype L1;
  subtype L2;
  subtype Linf;
  int n = 0;
  norms->getNorms(L1, L2, Linf, n);
  
  const Uintah::TypeDescription * td1 = field1.getTypeDescription();
  GridIterator iter = getIterator( td1, patch, false);
  
  ConsecutiveRangeSet matls1 = da1->queryMaterials(var, patch, timestep);
  
  da1->query(field1, var, matl, patch, timestep);

  map<const Patch*, Field*> patch_FieldMap;
  typename map<const Patch*, Field*>::iterator findIter;
  
  //__________________________________
  //  Loop over the cells of patch1
  for( ; !iter.done(); iter++ ) {
    IntVector c = *iter;
    
    const Patch* patch2 = cellToPatchMap[c];

    // find the number of occurances of the key (patch2) in the map
    int count = patch_FieldMap.count(patch2);
    
    if (count == 0 ) {                       // field is not in the map and has not been loaded previously
      field2 = scinew Field();
      
      patch_FieldMap[patch2] = field2;       // store value (field) in the map
      
      da2->query(*field2, var, matl, patch2, timestep);
      
    }else {                                  // field is in the map
      field2 = patch_FieldMap[patch2];       // pull out field
    }

    // bulletproofing
    Point p1 = patch->getCellPosition(c);
    Point p2 = patch2->getCellPosition(c);
    
    if (p1.x() != p2.x()   ||
       (p1.y() != p2.y() ) ||
       (p1.z() != p2.z() ) ){
      cout <<"\n__________________________________\n "
            << "You can't compare data at different physical locations  \n"
            << " uda1 data location: " << p1 << "\n uda2 data location: " << p2 << endl;
      abort_uncomparable();
    }
    
    // do the work
    n += 1;
    subtype diff = Abs(field1[c] - (*field2)[c]);
    L1 += diff;
    L2 += diff * diff;
    Linf = Max(Linf, diff);
  }
  
  norms->setNorms(L1, L2, Linf, n);
  
  // now cleanup memory.
  typename map<const Patch*, Field*>::iterator itr = patch_FieldMap.begin();
  for ( ; itr != patch_FieldMap.end(); itr++) {
    delete (*itr).second;
  }
}


//__________________________________
// At each cell index determine what patch you're on and store
// it in an cellToPatchMap
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

  // find the index range for this level and variable type basis
  IntVector bl=IntVector(0,0,0);
  IntVector low  = patches->get(0)->getExtraLowIndex(basis,bl);
  IntVector high = patches->get(0)->getExtraHighIndex(basis,bl);

  for (int i = 1; i < patches->size(); i++) {
    low  = Min(low,  patches->get(i)->getExtraLowIndex(basis,bl));
    high = Max(high, patches->get(i)->getExtraHighIndex(basis,bl));
  }
  
  // resize the array and initialize it
  patchMap.resize(low, high);
  patchMap.initialize(0);

  Level::const_patchIterator patch_iter;
  for(patch_iter = level->patchesBegin(); patch_iter != level->patchesEnd(); patch_iter++) {
    const Patch* patch = *patch_iter;
    
    IntVector ex_low = patch->getExtraLowIndex(basis,bl);
    IntVector ex_high = patch->getExtraHighIndex(basis,bl);
    
    ASSERT(Min(ex_low,  low) == low);
    ASSERT(Max(ex_high, high) == high);
    
    patchMap.rewindow(ex_low, ex_high);
    
    for (Array3<const Patch*>::iterator iter = patchMap.begin(); iter != patchMap.end(); iter++) {
      
      if (*iter != 0) {
       
       #if 0
       cerr << "Patches " << patch->getID() << " and " 
               << (*iter)->getID() << " overlap on the same file at time " << time
               << " in " << filebase << " at index " << iter.getIndex() << endl;
        #endif
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
        *iter = patch;     // insert patch into the array
      }
    }  //patchMap
  }  //level
  patchMap.rewindow(low, high);
}
//______________________________________________________________________
// create the directory   Lnorms/LevelIndex
void
createDirectory(string& levelIndex, string& path)
{
  string dirName = "./Lnorm";
  DIR *check = opendir(dirName.c_str());
  if ( check == NULL ) {
    cout << "Making directory "<< dirName<<endl;
    MKDIR( dirName.c_str(), 0777 );
  } else {
    closedir(check);
  }
  
  // level index
  path = dirName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == NULL ) {
    cout << "Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
//__________________________________
//
void createFile(string& filename, const int timestep)
{ 
  if(timestep == 0){
    ofstream out(filename.c_str(), ios_base::out);
    if(! out){
      cerr << " could not open output file: " << filename << endl;
      abort_uncomparable();
    }
    cout << " Now creating the file: "<< filename << endl;
    out << "#Time \t\t\t L1 \t\t L2 \t\t Linf" <<endl;
    out.close();
  }
}


//______________________________________________________________________
// Main
//______________________________________________________________________
int
main(int argc, char** argv)
{
  Thread::setDefaultAbortMode("exit");
  string ignoreVar = "none";
  string filebase1;
  string filebase2;

  //__________________________________
  // Parse Args:
  for( int i = 1; i < argc; i++ ) {
    string s = argv[i];
    if(s == "-ignoreVariable") {
      if (++i == argc){
        usage("-ignoreVariable, no variable given", argv[0]);
      }else{
        ignoreVar = argv[i];
      }
    }else if(s[0] == '-' && s[1] == 'h' ) { // lazy check for -h[elp] option
      usage( "", argv[0] );
    }else {
      if (filebase1 != "") {
        if (filebase2 != ""){
          usage(s, argv[0]);
        }else{
          filebase2 = argv[i];
        }
      }else{
        filebase1 = argv[i];
      }
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

  //__________________________________
  // bulletproofing   
  ASSERTEQ(vars.size(),  types.size());
  ASSERTEQ(vars2.size(), types2.size());

  if (vars.size() != vars2.size()) {
    cerr << filebase1 << " has " << vars.size() << " variables\n";
    cerr << filebase2 << " has " << vars2.size() << " variables\n";
    abort_uncomparable();
  }
 
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

  vector<int> index, index2;
  vector<double> times, times2;

  da1->queryTimesteps(index,  times);
  da2->queryTimesteps(index2, times2);

  ASSERTEQ(index.size(),  times.size());
  ASSERTEQ(index2.size(), times2.size());

  //__________________________________
  //  create the output directory
  GridP g        = da1->queryGrid(0);
  int numLevels  = g->numLevels();
  string path;
  for(int l=0;l<numLevels;l++){
    ostringstream li;
    li << "L-" << l;
    string levelIndex = li.str();
    createDirectory(levelIndex, path);
  }

  //__________________________________
  // Look over timesteps
  for(unsigned long t = 0; t < times.size() && t < times2.size(); t++){

    double time1 = times[t];
    double time2 = times2[t];
    cout << "time = " << time1 << "\n";
    GridP grid1  = da1->queryGrid(t);
    GridP grid2  = da2->queryGrid(t);

    BBox b1, b2;
    grid1->getInteriorSpatialRange(b1);
    grid2->getInteriorSpatialRange(b2);
    
    // warn the user that the computational domains are different
    if ((b1.min() != b2.min() ) ||
        (b1.max() != b2.max() ) ){
      cout << " The compuational domains of uda1 & uda2 are different" << endl;
      cout << " uda1: " << b1 << "\n uda2: " << b2 << endl;
    }
    
    // bullet proofing
    if (grid1->numLevels() != grid2->numLevels()) {
      cerr << "Grid at time " << time1 << " in " << filebase1 << " has " << grid1->numLevels() << " levels.\n";
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

      cout << "\t" << var << "\n";

      Patch::VariableBasis basis=Patch::translateTypeToBasis(td->getType(),false);
      //__________________________________
      //  loop over levels
      for(int l=0;l<grid1->numLevels();l++){
        cout << " \t\t L-" << l;
        LevelP level1  = grid1->getLevel(l);
        LevelP level2  = grid2->getLevel(l);

        //check patch coverage
        vector<Region> region1, region2, difference1, difference2;

        for(int i=0;i<level1->numPatches();i++){
          const Patch* patch=level1->getPatch(i);
          region1.push_back(Region(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex()));
        }

        for(int i=0;i<level2->numPatches();i++){
          const Patch* patch=level2->getPatch(i);
          region2.push_back(Region(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex()));
        }

        difference1 = Region::difference(region1,region2);
        difference2 = Region::difference(region1,region2);

        if(!difference1.empty() || !difference2.empty()){
          cerr << "\n__________________________________\n"
               << "The physical region covered on level " << l << " is not the same on both udas\n"
               << "If one of the udas has a smaller computational domain make sure it's the first\n"
               << "one listed in the command line arguments\n";
          abort_uncomparable();
        }

        // for each cell assign a patch to it on level1 and level2
        Array3<const Patch*> cellToPatchMap1;
        Array3<const Patch*> cellToPatchMap2;

        BuildCellToPatchMap(level1, filebase1, cellToPatchMap1,  time1, basis);
        BuildCellToPatchMap(level2, filebase2, cellToPatchMap2, time2, basis);

        // bulletproofing
        for ( Array3<const Patch*>::iterator nodePatchIter = cellToPatchMap1.begin(); nodePatchIter != cellToPatchMap1.end(); nodePatchIter++) {

          IntVector index = nodePatchIter.getIndex();

          if ((cellToPatchMap1[index] == 0 && cellToPatchMap2[index] != 0) ||
              (cellToPatchMap2[index] == 0 && cellToPatchMap1[index] != 0)) {
            cerr << "Inconsistent patch coverage on level " << l << " at time " << time1 << endl;

            if (cellToPatchMap1[index] != 0) {
              cerr << index << " is covered by " << filebase1 << " and not " << filebase2 << endl;
            } else {
              cerr << index << " is covered by " << filebase2 << " and not " << filebase1 << endl;
            }

            abort_uncomparable();
          }
        }

        //__________________________________
        //  compare the fields (CC, SFC(X,Y,Z), NC variables
        const Patch* dummyPatch=level1->getPatch(0);
        ConsecutiveRangeSet matls = da1->queryMaterials(var, dummyPatch, t);
        
        for (ConsecutiveRangeSet::iterator matlIter = matls.begin();matlIter != matls.end(); matlIter++){
          int matl = *matlIter;
          
          cout << "\t";  // output formatting
          if(matl> 0 ){
            cout << "\t\t";
          }
          cout << " Matl-"<<matl;
          ostringstream fname;
          fname<< path << "/" << var << "_" << matl;
          string filename = fname.str();
          createFile(filename, t);
          
          Norms<int>* inorm    = scinew Norms<int>();
          Norms<double>* dnorm = scinew Norms<double>();
          Norms<Vector>* vnorm = scinew Norms<Vector>();
          Vector zero = Vector(0,0,0);

          inorm->setNorms(0,0,0,0);
          dnorm->setNorms(0,0,0,0);
          vnorm->setNorms(zero,zero,zero,0);
                    
          Level::const_patchIterator iter;
          for(iter = level1->patchesBegin();iter != level1->patchesEnd(); iter++) {
            const Patch* patch = *iter;

            switch(td->getType()){
              //__________________________________
              //  CC
              case Uintah::TypeDescription::CCVariable:                                                   
                switch(subtype->getType()){                                                                 
                  case Uintah::TypeDescription::int_type:{                                                         
                    compareFields<CCVariable<int>,int>( inorm, da1, da2, var, matl, patch, cellToPatchMap2, t);        
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<CCVariable<double>,double>(dnorm,da1, da2, var, matl, patch, cellToPatchMap2, t); 
                    break;
                  }               
                  case Uintah::TypeDescription::Vector:{
                    compareFields<CCVariable<Vector>,Vector>(vnorm,da1, da2, var, matl, patch, cellToPatchMap2, t);  
                    break;
                  }
                  default:
                    cout << " Data type not supported "<< td->getName() <<  endl;
                }
                break;
              //__________________________________
              //  NC
              case Uintah::TypeDescription::NCVariable:                                                   
                switch(subtype->getType()){                                                                 
                  case Uintah::TypeDescription::int_type:{                                                         
                    compareFields<NCVariable<int>,int>(inorm, da1, da2, var, matl, patch, cellToPatchMap2, t);        
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<NCVariable<double>,double>(dnorm, da1, da2, var, matl, patch, cellToPatchMap2, t); 
                    break;
                  }               
                  case Uintah::TypeDescription::Vector:{
                    compareFields<NCVariable<Vector>,Vector>(vnorm, da1, da2, var, matl, patch, cellToPatchMap2, t);  
                    break;
                  }
                  default:
                    cout << " Data type not supported "<< td->getName() <<  endl;
                }
                break;
              //__________________________________
              //  SFCX
              case Uintah::TypeDescription::SFCXVariable:                                                   
                switch(subtype->getType()){                                                                 
                  case Uintah::TypeDescription::int_type:{                                                         
                    compareFields<SFCXVariable<int>,int>(inorm, da1, da2, var, matl, patch, cellToPatchMap2, t);        
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<SFCXVariable<double>,double>(dnorm, da1, da2, var, matl, patch, cellToPatchMap2, t); 
                    break;
                  }
                  default:
                    cout << " Data type not supported "<< td->getName() <<  endl;
                }
                break; 
              //__________________________________
              //  SFCY
              case Uintah::TypeDescription::SFCYVariable:                                                   
                switch(subtype->getType()){                                                                 
                  case Uintah::TypeDescription::int_type:{                                                         
                    compareFields<SFCYVariable<int>,int>(inorm, da1, da2, var, matl, patch, cellToPatchMap2, t);        
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<SFCYVariable<double>,double>(dnorm, da1, da2, var, matl, patch, cellToPatchMap2, t); 
                    break;
                  }
                  default:
                    cout << " Data type not supported "<< td->getName() <<  endl;
                }
                break;
              //__________________________________
              //  SFCZ
              case Uintah::TypeDescription::SFCZVariable:                                                   
                switch(subtype->getType()){                                                                 
                  case Uintah::TypeDescription::int_type:{                                                         
                    compareFields<SFCZVariable<int>,int>(inorm, da1, da2, var, matl, patch, cellToPatchMap2, t);        
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<SFCZVariable<double>,double>(dnorm, da1, da2, var, matl, patch, cellToPatchMap2, t); 
                    break;
                  }
                  default:
                    cout << " Data type not supported "<< td->getName() <<  endl;
                }
                break;
              default:
                cout << " Data type not yet supported: " << td->getName() << endl;
              break;
            }
          }  // patches
          
          // only one these conditionals true
          if (inorm->get_n() > 0){     
            inorm->printNorms(); 
            inorm->outputNorms(time1, filename);      
          }                            
          if (dnorm->get_n() > 0){     
            dnorm->printNorms();
            dnorm->outputNorms(time1, filename);      
          }                            
          if (vnorm->get_n() > 0){     
            vnorm->printNorms();
            vnorm->outputNorms(time1, filename);   
          }                            
          delete inorm;                
          delete dnorm;                
          delete vnorm;

        }  // matls
      }  // levels
    }  // variables
  }

  if (times.size() != times2.size()) {
    cout << endl;
    cout << filebase1 << " has " << times.size() << " timesteps\n";
    cout << filebase2 << " has " << times2.size() << " timesteps\n";
    abort_uncomparable();
  }
  delete da1;
  delete da2;
  return 0;
}

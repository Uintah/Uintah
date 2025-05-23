/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Parallel/Parallel.h>
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
  cout << "  -ignoreVariables [var1,var2....] (Skip these variables. Comma delimited list, no spaces.)\n";
  cout << "  -compareVariables[var1,var2....] (Only compare these variables. Comma delimited list, no spaces.)\n";
  cout << "  -timeTolerance   [double]        (The allowable difference of abs(uda1:time - uda2:time) when comparing each timestep.  Default is 1e-5)\n";
  cout << "  -h[elp]\n";
  Parallel::exitAll(1);
}
//__________________________________
void abort_uncomparable()
{
  cerr << "\nThe uda directories may not be compared.\n";
  Parallel::exitAll(5);
}

//__________________________________
//    Returns: sqrt( Vector )
Vector Sqrt(const Vector a)
{
  return Vector(Sqrt(a.x()), Sqrt(a.y()), Sqrt(a.z()));
}

//__________________________________
//     Returns: sqrt( stencil7 )
inline Stencil7 Sqrt(const Stencil7 a)
{
  Stencil7 me;

  for(int i = 0; i<7; i++){
    me[i] = sqrt( a[i] );
  }
  return me;
}
//__________________________________
//    Returns: stencil7 * stencil7
inline Stencil7 operator*(const Stencil7& a, const Stencil7& b)
{
  Stencil7 me;

  for(int i = 0; i<7; i++){
    me[i] = a[i] *b[i];
  }
  return me;
}
//__________________________________
//    Returns: stencil7 / stencil7
inline Stencil7 operator/(const Stencil7& a, const Stencil7& b)
{
  Stencil7 me;

  for(int i = 0; i<7; i++){
    me[i] = a[i] / b[i];
  }
  return me;
}
//__________________________________
//    Returns: stencil7 - stencil7
inline Stencil7 operator-(const Stencil7& a, const Stencil7& b)
{
  Stencil7 me;

  for(int i = 0; i<7; i++){
    me[i] = a[i] - b[i];
  }
  return me;
}
//__________________________________
//    Returns: stencil7 + stencil7
inline Stencil7 operator+(const Stencil7& a, const Stencil7& b)
{
  Stencil7 me;

  for(int i = 0; i<7; i++){
    me[i] = a[i] + b[i];
  }
  return me;
}
//__________________________________
//    Returns: Max(stencil7)
inline Stencil7 Max(const Stencil7& a, const Stencil7& b)
{
  Stencil7 me;

  for(int i = 0; i<7; i++){
    me[i] = Max( a[i], b[i] );
  }
  return me;
}
//__________________________________
//    Returns: Abs(stencil7)
inline Stencil7 Abs(const Stencil7& a)
{
  Stencil7 me;

  for(int i = 0; i<7; i++){
    me[i] = std::abs( a[i] );
  }
  return me;
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

    void printNorms()             // integers, doubles, Vector
    {
      T denom(d_n);
      d_L1   = d_L1/denom ;
      d_L2   = Sqrt( d_L2/denom );
      cout << " \t norms: L1 " << d_L1 << " L2: " << d_L2 << " Linf: " << d_Linf<< " n_cells: " << d_n<<endl;
    }

    void printNormsS7()           // Stencil7 Variables
    {
      T denom;
      denom.initialize( d_n );
      d_L1   = d_L1/denom ;
      d_L2   = Sqrt( d_L2/denom );
      cout << " \t norms: L1    " << d_L1 << endl;
      cout << " \t        L2:   " << d_L2 << endl;
      cout << " \t        Linf: " << d_Linf<< endl;
      cout << " \t        n_cells: " << d_n<<endl;
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
                   const Patch* patch1,
                   const Array3<const Patch*>& cellToPatchMap2,
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
  GridIterator iter = getIterator( td1, patch1, false);

  ConsecutiveRangeSet matls1 = da1->queryMaterials(var, patch1, timestep);

  da1->query(field1, var, matl, patch1, timestep);

  map<const Patch*, Field*> patch2_FieldMap;
  typename map<const Patch*, Field*>::iterator findIter;

  //__________________________________
  //  Loop over the cells of patch1
  for( ; !iter.done(); iter++ ) {
    IntVector c = *iter;

    const Patch* patch2 = cellToPatchMap2.get(c);

    // find the number of occurances of the key (patch2) in the map
    int count = patch2_FieldMap.count(patch2);

    if (count == 0 ) {                       // field is not in the map and has not been loaded previously
      field2 = scinew Field();

      patch2_FieldMap[patch2] = field2;       // store value (field) in the map

      da2->query(*field2, var, matl, patch2, timestep);

    }else {                                  // field is in the map
      field2 = patch2_FieldMap[patch2];       // pull out field
    }

    // bulletproofing
    Point p1 = patch1->getCellPosition(c);
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
    L1 = L1 + diff;
    L2 = L2 + (diff * diff);
    Linf = Max(Linf, diff);
  }

  norms->setNorms(L1, L2, Linf, n);

  // now cleanup memory.
  typename map<const Patch*, Field*>::iterator itr = patch2_FieldMap.begin();
  for ( ; itr != patch2_FieldMap.end(); itr++) {
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

  Level::const_patch_iterator patch_iter;
  for(patch_iter = level->patchesBegin(); patch_iter != level->patchesEnd(); patch_iter++) {
    const Patch* patch = *patch_iter;

    IntVector ex_low = patch->getExtraLowIndex(basis,bl);
    IntVector ex_high = patch->getExtraHighIndex(basis,bl);

    ASSERT(Min(ex_low,  low) == low);
    ASSERT(Max(ex_high, high) == high);

    patchMap.rewindow(ex_low, ex_high);

    serial_for( patchMap.range(), [&](int i, int j, int k) {
      if (patchMap.get(i,j,k)) {
        // in some cases, we can have overlapping patches, where an extra cell/node
        // overlaps an interior cell/node of another patch.  We prefer the interior
        // one.  (if there are two overlapping interior ones (nodes or face centers only),
        // they should have the same value.  However, since this patchMap is also used
        // for cell centered variables give priority to the patch that has this index within
        // its interior cell centered variables
        IntVector in_low  = patch->getLowIndex(basis);
        IntVector in_high = patch->getHighIndex(basis);
        if (   i >= in_low[0] && j >= in_low[1] && k >= in_low[2]
            && i < in_high[0] && j < in_high[1] && k < in_high[2] )
        {
          patchMap.get(i,j,k) = patch;
        }
      }else{
        patchMap.get(i,j,k) = patch;     // insert patch into the array
      }
    });  //patchMap
  } //level
  patchMap.rewindow(low, high);
}
//______________________________________________________________________
// create the directory   Lnorms/LevelIndex
void
createDirectory(string& levelIndex, string& path)
{
  string dirName = "./Lnorm";
  DIR *check = opendir(dirName.c_str());
  if ( check == nullptr ) {
    cout << "Making directory "<< dirName<<endl;
    MKDIR( dirName.c_str(), 0777 );
  } else {
    closedir(check);
  }

  // level index
  path = dirName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == nullptr ) {
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
//  parse user input and create a vector of strings.  Deliminiter is ","
vector<string>  parseVector( const char* input)
{
  vector<string> result;
  stringstream ss (input);  
  
  while( ss.good() ){
    string substr;
    getline( ss, substr, ',' );
    result.push_back( substr );
  }
  return result;
}
//______________________________________________________________________
// Main
//______________________________________________________________________
int
main(int argc, char** argv)
{
  Uintah::Parallel::initializeManager(argc, argv);

  vector<string> ignoreVars;
  vector<string> compareVars;
  string filebase1;
  string filebase2;
  double timeTolerance = 1e-5;

  //__________________________________
  // Parse Args:
  for( int i = 1; i < argc; i++ ) {
    string s = argv[i];
    
    if(s == "-ignoreVariables") {
      if (++i == argc){
        usage("-ignoreVariables, no variable given", argv[0]);
      }
      else{
        ignoreVars = parseVector( argv[i] );
      }
    }
    else if(s == "-compareVariables") {
      if (++i == argc){
        usage("-compareVariables, no variable given", argv[0]);
      }
      else{
        compareVars = parseVector( argv[i] );
      }
    }
    else if(s == "-timeTolerance") {
      if (++i == argc){
        usage("-timeTolerance, no variable given", argv[0]);
      }
      else{
        timeTolerance = std::stod( argv[i]) ;
      }
    }
    else if(s[0] == '-' && s[1] == 'h' ) { // lazy check for -h[elp] option
      usage( "", argv[0] );
    }
    else {
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
  vector<int>    num_matls, num_matls2;
  vector<const Uintah::TypeDescription*> types, types2;
  typedef vector< pair<string, const Uintah::TypeDescription*> > VarTypeVec;
  VarTypeVec vartypes1;
  VarTypeVec vartypes2;

  da1->queryVariables( vars,  num_matls,  types );
  da2->queryVariables( vars2, num_matls2, types2 );

  vartypes1.resize( vars.size() );
  vartypes2.resize( vars2.size() );
    
  //__________________________________
  // Create a list of variables minus the ignored variables
  // uda 1
  for (auto i = ignoreVars.begin(); i != ignoreVars.end(); i++){
    cout << "Ignoring variable: " << *i << endl;
  }

  int count = 0;
  for (unsigned int i = 0; i < vars.size(); i++) {
    auto me = find( ignoreVars.begin(), ignoreVars.end(), vars[i] );
    // if vars[i] is NOT in the ignore Variables list make a pair
    if (me == ignoreVars.end()){
      vartypes1[count] = make_pair(vars[i], types[i]);
      count ++;
    }
  }
  vars.resize(count);
  vartypes1.resize(vars.size());

  // uda 2
  count =0;
  for (unsigned int i = 0; i < vars2.size(); i++) {
    auto me = find( ignoreVars.begin(), ignoreVars.end(), vars2[i] );

    if (me == ignoreVars.end()){
      vartypes2[count] = make_pair(vars2[i], types2[i]);
      count ++;
    }
  }
  vars2.resize(count);
  vartypes2.resize(vars2.size());
  
  //__________________________________
  // Create a list of variables to compare if the user wants
  // specifies a few variables.  Default is to compare all

  if( compareVars.size() > 0 ){
    for (auto i = compareVars.begin(); i != compareVars.end(); i++){
      cout << "Variable: " << *i << endl;
    }

    // uda 1
    count = 0;
    for (unsigned int i = 0; i < vars.size(); i++) {
      auto me = find(compareVars.begin(),compareVars.end(),vars[i]);

      if (me != compareVars.end()){
        vartypes1[count] = make_pair(vars[i], types[i]);
        count ++;
      }
    }
    vars.resize(count);
    vartypes1.resize(vars.size());

    // uda 2
    count =0;
    for (unsigned int i = 0; i < vars2.size(); i++) {
      auto me = find( compareVars.begin(), compareVars.end(), vars2[i] );

      if (me != compareVars.end()){
        vartypes2[count] = make_pair(vars2[i], types2[i]);
        count ++;
      }
    }
    vars2.resize(count);
    vartypes2.resize(vars2.size());    
  }

  size_t vars1_size = vars.size();    // needed for bullet proofing
  size_t vars2_size = vars2.size();
  
  
  //__________________________________
  //  created vector of vars to compare
  bool do_udas_have_same_nVars = true;

  if ( vartypes1.size() == vartypes2.size() )  {
    for (unsigned int i = 0; i < vars.size(); i++) {
      vars[i]   = vartypes1[i].first;
      types[i]  = vartypes1[i].second;
      vars2[i]  = vartypes2[i].first;
      types2[i] = vartypes2[i].second;
    }    
  }

  //__________________________________
  // If the number of variables in each uda
  // differs then find a common set of variables
  if(vars1_size != vars2_size ){

    do_udas_have_same_nVars = false;

    for (unsigned int i = 0; i < vars.size(); i++) {
      vartypes1[i] = make_pair(vars[i], types[i]);
    }

    for (unsigned int i = 0; i < vars2.size(); i++) {
      vartypes2[i] = make_pair(vars2[i], types2[i]);
    }

    cerr << "\nWARNING: The udas contain a different number of variables.  Now comparing the common set of variables.\n";

    VarTypeVec commonVars;     // common variables

    set_intersection(vartypes1.begin(), vartypes1.end(),
                     vartypes2.begin(), vartypes2.end(),
                     std::back_inserter(commonVars) );

    size_t size = commonVars.size();
    vars.resize(size);
    vars2.resize(size);
    vartypes1.resize(size);
    vartypes2.resize(size);

    for (unsigned int i = 0; i <size; i++) {
      vars[i]   = commonVars[i].first;
      types[i]  = commonVars[i].second;
      vars2[i]  = commonVars[i].first;
      types2[i] = commonVars[i].second;
    }
  }

  //__________________________________
  // bulletproofing
  ASSERTEQ(vars.size(),  types.size());
  ASSERTEQ(vars2.size(), types2.size());

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
  for(unsigned long tstep = 0; tstep < times.size() && tstep < times2.size(); tstep++){

    double time1 = times[tstep];
    double time2 = times2[tstep];
    double diffTime = abs(times[tstep] - times2[tstep]);

    cout << "time = " << time1 << "     Difference in output time: abs(uda1:time - uda2:times) = "<< diffTime<< "\n";
    GridP grid1  = da1->queryGrid(tstep);
    GridP grid2  = da2->queryGrid(tstep);

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

    if ( diffTime > timeTolerance) {
      cerr << "Output time " << times[tstep] << " in " << filebase1 << " differs from the\n";
      cerr << "output time " << times2[tstep] << " in " << filebase2 << " exceeding the allowable tolerance " << timeTolerance << "\n";
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

        //__________________________________
        //  bulletproofing does the variable exist in both DAs on this timestep?
        //  This problem mainly occurs if <outputInitTimestep> has been specified.
        bool existsDA1 = true;
        bool existsDA2 = true;
        Level::const_patch_iterator iter;
        for(iter = level1->patchesBegin(); iter != level1->patchesEnd(); iter++) {
          const Patch* patch = *iter;
          if ( ! da1->exists( var, patch, tstep ) ){
            existsDA1 = false;
          }
        }
        for(iter = level2->patchesBegin(); iter != level2->patchesEnd(); iter++) {
          const Patch* patch = *iter;
          if ( ! da2->exists( var, patch, tstep ) ){
            existsDA2 = false;
          }
        }
        if( existsDA1 != existsDA2 ) {
          cerr << "\nThe variable ("<< var << ") was not found on timestep (" << tstep  <<  ") in both udas. \n"
               << " uda1: " << existsDA1 << " uda2: " << existsDA2 << "\n"
               << "If this occurs on timestep 0 then ("<< var<< ") was not computed in the initialization task."<<endl;
          abort_uncomparable();
        }

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

        serial_for( cellToPatchMap1.range(), [&](int i, int j, int k) {
          if ((cellToPatchMap1.get(i,j,k) == nullptr && cellToPatchMap2.get(i,j,k) != nullptr) ||
              (cellToPatchMap2.get(i,j,k) == nullptr && cellToPatchMap1.get(i,j,k) != nullptr)) {
            cerr << "Inconsistent patch coverage on level " << l << " at time " << time1 << endl;
            if (cellToPatchMap1.get(i,j,k) != nullptr) {
              cerr << "(" << i << "," << j << "," << k << ")" << " is covered by " << filebase1 << " and not " << filebase2 << endl;
            }
            else {
              cerr << "(" << i << "," << j << "," << k << ")" << " is covered by " << filebase2 << " and not " << filebase1 << endl;
            }
            abort_uncomparable();
          }
        });

        //__________________________________
        //  compare the fields (CC, SFC(X,Y,Z), NC variables
        const Patch* dummyPatch=level1->getPatch(0);
        ConsecutiveRangeSet matls = da1->queryMaterials(var, dummyPatch, tstep);

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
          createFile(filename, tstep);

          Norms<int>* inorm       = scinew Norms<int>();
          Norms<double>* dnorm    = scinew Norms<double>();
          Norms<float>*  fnorm    = scinew Norms<float>();
          Norms<Vector>* vnorm    = scinew Norms<Vector>();
          Norms<Stencil7>* s7norm = scinew Norms<Stencil7>();

          Vector zeroV = Vector(0,0,0);
          Stencil7 zeroS7;
          zeroS7.initialize(0.0);

          inorm->setNorms(0, 0, 0, 0);
          dnorm->setNorms(0, 0, 0, 0);
          fnorm->setNorms(0, 0, 0, 0);
          vnorm->setNorms(  zeroV,  zeroV,  zeroV,  0);
          s7norm->setNorms( zeroS7, zeroS7, zeroS7, 0 );

          Level::const_patch_iterator iter;
          for(iter = level1->patchesBegin();iter != level1->patchesEnd(); iter++) {
            const Patch* patch1 = *iter;

            switch(td->getType()){
              //__________________________________
              //  CC
              case Uintah::TypeDescription::CCVariable:
                switch(subtype->getType()){
                  case Uintah::TypeDescription::int_type:{
                    compareFields<CCVariable<int>,int>( inorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::float_type:{
                    compareFields<CCVariable<float>,float>(fnorm,da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<CCVariable<double>,double>(dnorm,da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::Vector:{
                    compareFields<CCVariable<Vector>,Vector>(vnorm,da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::Stencil7:{
                    compareFields<CCVariable<Stencil7>,Stencil7>(s7norm,da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
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
                    compareFields<NCVariable<int>,int>(inorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::float_type:{
                    compareFields<NCVariable<float>,float>(fnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<NCVariable<double>,double>(dnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::Vector:{
                    compareFields<NCVariable<Vector>,Vector>(vnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
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
                    compareFields<SFCXVariable<int>,int>(inorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::float_type:{
                    compareFields<SFCXVariable<float>,float>(fnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<SFCXVariable<double>,double>(dnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
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
                    compareFields<SFCYVariable<int>,int>(inorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::float_type:{
                    compareFields<SFCYVariable<float>,float>(fnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<SFCYVariable<double>,double>(dnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
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
                    compareFields<SFCZVariable<int>,int>(inorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::float_type:{
                    compareFields<SFCZVariable<float>,float>(fnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
                    break;
                  }
                  case Uintah::TypeDescription::double_type:{
                    compareFields<SFCZVariable<double>,double>(dnorm, da1, da2, var, matl, patch1, cellToPatchMap2, tstep);
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
          if (inorm->get_n() > 0){        // integers
            inorm->printNorms();
            inorm->outputNorms(time1, filename);
          }
          if (fnorm->get_n() > 0){        // floats
            fnorm->printNorms();
            fnorm->outputNorms(time1, filename);
          }
          if (dnorm->get_n() > 0){        // doubles
            dnorm->printNorms();
            dnorm->outputNorms(time1, filename);
          }
          if (vnorm->get_n() > 0){        //Vector
            vnorm->printNorms();
            vnorm->outputNorms(time1, filename);
          }
          if (s7norm->get_n() > 0){       //Stencil7
            s7norm->printNormsS7();
            s7norm->outputNorms(time1, filename);
          }
          delete inorm;
          delete fnorm;
          delete dnorm;
          delete vnorm;
          delete s7norm;

        }  // matls
      }  // levels
    }  // variables
  }

  //__________________________________
  //
  if ( ! do_udas_have_same_nVars ) {
    cout << "\n__________________________________\n\n";
    cout << "  ERROR:  the udas don't have the same number of variables\n";
    cout << "    " << filebase1 << " has " << vars1_size << " variables\n";
    cout << "    " << filebase2 << " has " << vars2_size << " variables\n";
    abort_uncomparable();
  }

  if (times.size() != times2.size()) {
    cout << "\n__________________________________\n\n";
    cout << "  ERROR:  the udas don't have the same number timesteps.\n";
    cout << "    " << filebase1 << " has " << times.size() << " timesteps\n";
    cout << "    " << filebase2 << " has " << times2.size() << " timesteps\n";
    abort_uncomparable();
  }
  delete da1;
  delete da2;
  return 0;
}

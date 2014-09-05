/*
 *  puda.cc: Print out a uintah data archive
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 *  Copyright (C) 2000 U of U
 */

/*
 *  Support for printing out Tecplot data added by Patric Hu
 *  Department of Mechanical Enginerring, U of U, 2003.
 *
 *  Currently it only supports CCVariables.
 * 
 * Usage of converting Uintah data archive to a tecplot data file
 * puda -tecplot <i_xd> <uda directory> :
 *       print all CCVariables into different tecplot data files 
 * puda -tecplot <i_xd> <CCVariable's Name> <uda directory>:
 *       print one CCVariable into a tecplot data file
 * puda -tecplot <i_xd> <tskip> <uda directory>:
 *       print all CCVariables into different tecplot data files
 *       by every tskip time steps
 * puda -tecplot <i_xd> <CCVariable's Name> <tskip> <uda directory>:
 *       print one CCVariable into a tecplot data file
 *       by every tskip time steps
 * i_xd may be i_1d, i_2d, i_3d for 1D, 2D and 3D problem
 *
 */

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/Array3.h>

#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <stdio.h>
#include <math.h>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

typedef struct{
  vector<ShareAssignParticleVariable<double> > pv_double_list;
  vector<ShareAssignParticleVariable<float> > pv_float_list;
  vector<ShareAssignParticleVariable<Point> > pv_point_list;
  vector<ShareAssignParticleVariable<Vector> > pv_vector_list;
  vector<ShareAssignParticleVariable<Matrix3> > pv_matrix3_list;
  ShareAssignParticleVariable<Point> p_x;
} MaterialData;

// pre-declaration
void printParticleVariable(DataArchive* da, 
			   string particleVariable,
			   unsigned long time_step_lower,
			   unsigned long time_step_upper);

// takes a string and replaces all occurances of old with newch
string replaceChar(string s, char old, char newch) {
  string result;
  for (int i = 0; i<(int)s.size(); i++)
    if (s[i] == old)
      result += newch;
    else
      result += s[i];
  return result;
}

// use this function to open a pair of files for outputing
// data to the reat-time raytracer.
//
// in: pointers to the pointer to the files data and header
//     the file names
// out: inialized files for writing
//      boolean reporting the success of the file creation

bool setupOutFiles(FILE** data, FILE** header, string name, string head) {
  FILE* datafile;
  FILE* headerfile;
  string headername = name + string(".") + head;

  datafile = fopen(name.c_str(),"w");
  if (!datafile) {
    cerr << "Can't open output file " << name << endl;
    return false;
  }
  
  headerfile = fopen(headername.c_str(),"w");
  if (!headerfile) {
    cerr << "Can't open output file " << headername << endl;
    return false;
  }
  
  *data = datafile;
  *header = headerfile;
  return true;
}
//__________________________________
void findTimestep_loopLimits(const bool tslow_set, 
                             const bool tsup_set,
                             const vector<double> times,
                             unsigned long & time_step_lower,
                             unsigned long & time_step_upper)
{
  if (!tslow_set){
   time_step_lower =0;
  }else if (time_step_lower >= times.size()) {
   cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
   abort();
  }
  if (!tsup_set){
   time_step_upper = times.size()-1;
  }else if (time_step_upper >= times.size()) {
   cerr << "timestephigh must be between 0 and " << times.size()-1 << endl;
   abort();
  }
}

// given the various parts of the name we piece together the full name
string makeFileName(string raydatadir, string variable_file, string time_file, 
		    string patchID_file, string materialType_file) {

  string raydatafile;
  if (raydatadir != "")
    raydatafile+= raydatadir + string("/");
  raydatafile+= string("TS_") + time_file + string("/");
  if (variable_file != "")
    raydatafile+= string("VAR_") + variable_file + string(".");
  if (materialType_file != "")
    raydatafile+= string("MT_") + materialType_file + string(".");
  raydatafile+= string("PI_") + patchID_file;
  return raydatafile;
}

void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h[elp]\n";
  cerr << "  -timesteps\n";
  cerr << "  -gridstats\n";
  cerr << "  -listvariables\n";
  cerr << "  -varsummary\n";
  cerr << "  -jim1\n";
  cerr << "  -partvar <variable name>\n";
  cerr << "  -asci\n";
  cerr << "  -tecplot <variable name>\n";
  cerr << "  -cell_stresses\n";
  cerr << "  -rtdata [output directory]\n";
  cerr << "  -PTvar\n";
  cerr << "  -ptonly (prints out only the point location\n";
  cerr << "  -patch (outputs patch id with data)\n";
  cerr << "  -material (outputs material number with data)\n";
  cerr << "  -NCvar [double or float or point or vector]\n";
  cerr << "  -CCvar [double or float or point or vector]\n";
  cerr << "  -verbose (prints status of output)\n";
  cerr << "  -timesteplow [int] (only outputs timestep from int)\n";
  cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
  cerr << "*NOTE* to use -PTvar or -NVvar -rtdata must be used\n";
  cerr << "*NOTE* ptonly, patch, material, timesteplow, timestephigh "
       << "are used in conjuntion with -PTvar.\n\n";
    
  cerr << "USAGE IS NOT FINISHED\n\n";
  exit(1);
}

int main(int argc, char** argv)
{
  if (argc <= 1)
    // Print out the usage and die
    usage("", argv[0]);

  /*
   * Default values
   */
  bool do_timesteps=false;
  bool do_gridstats=false;
  bool do_listvars=false;
  bool do_varsummary=false;
  bool do_jim1=false;
  bool do_partvar=false;
  bool do_asci=false;
  bool do_cell_stresses=false;
  bool do_rtdata = false;
  bool do_NCvar_double = false;
  bool do_NCvar_float = false;
  bool do_NCvar_point = false;
  bool do_NCvar_vector = false;
  bool do_NCvar_matrix3 = false;
  bool do_CCvar_double = false;
  bool do_CCvar_float = false;
  bool do_CCvar_point = false;
  bool do_CCvar_vector = false;
  bool do_CCvar_matrix3 = false;
  bool do_PTvar = false;
  bool do_PTvar_all = true;
  bool do_patch = false;
  bool do_material = false;
  bool do_verbose = false;
  bool do_tecplot = false;
  bool do_all_ccvars = false;
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = 1;
  unsigned long time_step_inc = 1;
  bool tslow_set = false;
  bool tsup_set = false;
  int tskip = 1;
  string i_xd;
  string filebase;
  string raydatadir;
  string particleVariable;
  string ccVarInput;
  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-timesteps"){
      do_timesteps=true;}
    else if(s == "-tecplot"){ 
      do_tecplot = true;
      if(argc == 4) {
	do_all_ccvars = true;
	i_xd = argv[i+1];
	tskip = 1;
      } else if(argc == 5){
	do_all_ccvars = true;
	i_xd = argv[i+1];
	tskip = atoi(argv[i+2]);
      } else if(argc == 6 ) {
	i_xd = argv[i+1];
	ccVarInput = argv[i+2];
	tskip = atoi(argv[i+3]);
	if (ccVarInput[0] == '-')
	  usage("-tecplot <i_xd> <ccVariable name> <tskip> ", argv[0]);
      }
    } else if(s == "-gridstats"){
      do_gridstats=true;
    } else if(s == "-listvariables"){
      do_listvars=true;
    } else if(s == "-varsummary"){
      do_varsummary=true;
    } else if(s == "-jim1"){
      do_jim1=true;
    } else if(s == "-partvar"){
      do_partvar=true;
      particleVariable = argv[++i]; 
      if (particleVariable[0] == '-') 
        usage("-partvar <particle variable name>", argv[0]);
    } else if(s == "-asci"){
      do_asci=true;
    } else if(s == "-cell_stresses"){
      do_cell_stresses=true;
    } else if(s == "-rtdata") {
      do_rtdata = true;
      if (++i < argc) {
	s = argv[i];
	if (s[0] == '-')
	  usage("-rtdata", argv[0]);
	raydatadir = s;
      }
    } else if(s == "-NCvar") {
      if (++i < argc) {
	s = argv[i];
	if (s == "double")
	  do_NCvar_double = true;
	else if (s == "float")
	  do_NCvar_float = true;
	else if (s == "point")
	  do_NCvar_point = true;
	else if (s == "vector")
	  do_NCvar_vector = true;
	else if (s == "matrix3")
	  do_NCvar_matrix3 = true;
	else
	  usage("-NCvar", argv[0]);
      }
      else
	usage("-NCvar", argv[0]);
    } else if(s == "-CCvar") {
      if (++i < argc) {
	s = argv[i];
	if (s == "double")
	  do_CCvar_double = true;
	else if (s == "float")
	  do_CCvar_float = true;
	else if (s == "point")
	  do_CCvar_point = true;
	else if (s == "vector")
	  do_CCvar_vector = true;
	else if (s == "matrix3")
	  do_CCvar_matrix3 = true;
	else
	  usage("-CCvar", argv[0]);
      }
      else
	usage("-CCvar", argv[0]);
    } else if(s == "-PTvar") {
      do_PTvar = true;
    } else if (s == "-ptonly") {
      do_PTvar_all = false;
    } else if (s == "-patch") {
      do_patch = true;
    } else if (s == "-material") {
      do_material = true;
    } else if (s == "-verbose") {
      do_verbose = true;
    } else if (s == "-timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
      tslow_set = true;
    } else if (s == "-timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
      tsup_set = true;
    } else if (s == "-timestepinc") {
      time_step_inc = strtoul(argv[++i],(char**)NULL,10);
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
    } else if( filebase == "") {
      filebase = argv[i];
    } else {
      usage( s, argv[0]);
    }
  }

  if(filebase == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    DataArchive* da = scinew DataArchive(filebase);
    
    if(do_timesteps){
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";
      for(int i=0;i<(int)index.size();i++)
	cout << index[i] << ": " << times[i] << endl;
    }
    //______________________________________________________________________
    //    DO GRIDSTATS
    if(do_gridstats){
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      
      findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
           
      for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
	 double time = times[t];
        cout << "__________________________________"<<endl;
	 cout << "Timestep " << t << ": " << time << endl;
	 GridP grid = da->queryGrid(time);
	 grid->performConsistencyCheck();
	 grid->printStatistics();

        for(int l=0;l<grid->numLevels();l++){
	    LevelP level = grid->getLevel(l);
           cout << "Level: index " << level->getIndex() << ", id " << level->getID() << endl;

          for(Level::const_patchIterator iter = level->patchesBegin();
	         iter != level->patchesEnd(); iter++){
	     const Patch* patch = *iter;
	     cout << *patch << endl;
            cout << "\t   BC types: x- " << patch->getBCType(Patch::xminus) << ", x+ "<<patch->getBCType(Patch::xplus)
                 << ", y- "<< patch->getBCType(Patch::yminus) << ", y+ "<< patch->getBCType(Patch::yplus)
                 << ", z- "<< patch->getBCType(Patch::zminus) << ", z+ "<< patch->getBCType(Patch::zplus)<< endl;
          }
        }
      }
    }
    if(do_listvars){
      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      cout << "There are " << vars.size() << " variables:\n";
      for(int i=0;i<(int)vars.size();i++){
	cout << vars[i] << ": " << types[i]->getName() << endl;
      }
    }

    // Print a particular particle variable
    if (do_partvar) {
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      if (!tslow_set)
	time_step_lower =0;
      else if (time_step_lower >= times.size()) {
	cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
	abort();
      }
      if (!tsup_set)
	time_step_upper = times.size()-1;
      else if (time_step_upper >= times.size()) {
	cerr << "timestephigh must be between 0 and " << times.size()-1 << endl;
	abort();
      }
      printParticleVariable(da, particleVariable, time_step_lower, 
                            time_step_upper);
    }
    //______________________________________________________________________
    //            DO_TECPLOT
    if(do_tecplot){ // if begin: 1
      string ccVariable;
      bool ccVarFound = false;
      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      ASSERTEQ(vars.size(), types.size());
      cout << "There are " << vars.size() << " variables:\n";

      const Uintah::TypeDescription* td;
      const Uintah::TypeDescription* subtype;
      if(!do_all_ccvars) {
	for(int i=0;i<(int)vars.size();i++){
	  cout << vars[i] << ": " << types[i]->getName() << endl;
	  if(vars[i] == ccVarInput) {
	    ccVarFound = true;
	  }
	}
	if(!ccVarFound) {
	  cerr << "the input ccVariable for tecplot is not storaged in the Dada Archive" << endl;
	  abort();
	}
      }//end of (!do_all_ccvars)

      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";
      for(int i=0;i<(int)index.size();i++)
	cout << index[i] << ": " << times[i] << endl;

      if (!tslow_set)
	time_step_lower =0;
      else if (time_step_lower >= times.size()) {
	cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
	abort();
      }
      if (!tsup_set)
	time_step_upper = times.size()-1;
      else if (time_step_upper >= times.size()) {
	cerr << "timestephigh must be between 0 and " << times.size()-1 << endl;
	abort();
      }
              
      for(int i=0;i<(int)vars.size();i++){ //for loop over all the variables: 2
	cout << vars[i] << ": " << types[i]->getName() << endl;
	if(do_all_ccvars || ((!do_all_ccvars) && (vars[i] == ccVarInput))){ // check if do all CCVariables 
	  // or just do one variable: 3  
	  td = types[i];
	  subtype = td->getSubType();
	  ccVariable = vars[i];
	  switch(td->getType()){ //switch to get data type: 4
    
	    //___________________________________________
	    //  Curently C C  V A R I A B L E S Only
	    //
	  case Uintah::TypeDescription::CCVariable:
	    { //CCVariable case: 5 
	      //    generate the name of the output file;
	      string filename;
	      string fileroot("tec.");
	      string filetype(".dat");
	      filename = fileroot + ccVariable;
	      filename = filename + filetype;
	      ofstream outfile(filename.c_str());
	      outfile.setf(ios::scientific,ios::floatfield);
	      outfile.precision(20);

	      //    print out the Title of the output file according to the subtype of the CCVariables 
		       
	      outfile << "TITLE = " << "\"" << ccVariable << " tecplot data file" << "\"" << endl;

              if(i_xd == "i_3d") {
	        if(subtype->getType() == Uintah::TypeDescription::double_type) {
		  outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", "
			  << "\"" << ccVariable << "\""; 
	        }
	        if(subtype->getType() == Uintah::TypeDescription::float_type) {
		  outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", "
			  << "\"" << ccVariable << "\""; 
	        }
	        if(subtype->getType() == Uintah::TypeDescription::Vector || subtype->getType() == Uintah::TypeDescription::Point) {
		  outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", "
			  << "\"" << ccVariable << ".X" << "\", " << "\"" << ccVariable << ".Y" << "\", " 
			  << "\"" << ccVariable << ".Z" << "\"";
	        }
	        if(subtype->getType() == Uintah::TypeDescription::Matrix3) {
		  outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", " << "\"Z" << "\", " 
			  << "\"" << ccVariable  << ".1.1\"" << ", \"" << ccVariable << ".1.2\"" << ", \"" << ccVariable << ".1.3\""
			  << ", \"" << ccVariable << ".2.1\"" << ", \"" << ccVariable << ".2.2\"" << ", \"" << ccVariable << ".2.3\""
			  << ", \"" << ccVariable << ".3.1\"" << ", \"" << ccVariable << ".3.2\"" << ", \"" << ccVariable << ".3.3\"";
	        }
	        outfile << endl;
	      } else if(i_xd == "i_2d") {
	        if(subtype->getType() == Uintah::TypeDescription::double_type) {
		  outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " 
			  << "\"" << ccVariable << "\""; 
	        }
	        if(subtype->getType() == Uintah::TypeDescription::float_type) {
		  outfile << "VARIABLES = " << "\"X" << "\", " << "\"Y" << "\", " 
			  << "\"" << ccVariable << "\""; 
	        }
	        if(subtype->getType() == Uintah::TypeDescription::Vector || subtype->getType() == Uintah::TypeDescription::Point) {
		  outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", "
			  << "\"" << ccVariable << ".X" << "\", " << "\"" << ccVariable << ".Y" << "\"";
	        }
	        if(subtype->getType() == Uintah::TypeDescription::Matrix3) {
		  outfile << "VARIABLES =" << "\"X" << "\", " << "\"Y" << "\", " 
			  << "\"" << ccVariable  << ".1.1\"" << ", \"" << ccVariable << ".1.2\"" 
			  << ", \"" << ccVariable << ".2.1\"" << ", \"" << ccVariable << ".2.2\""; 
	        }
	        outfile << endl;
	      } else if(i_xd == "i_1d") {
	        if(subtype->getType() == Uintah::TypeDescription::double_type) {
		  outfile << "VARIABLES = " << "\"X" << "\", " << "\"" << ccVariable << "\""; 
	        }
	        if(subtype->getType() == Uintah::TypeDescription::float_type) {
		  outfile << "VARIABLES = " << "\"X" << "\", " << "\"" << ccVariable << "\""; 
	        }
	        if(subtype->getType() == Uintah::TypeDescription::Vector || subtype->getType() == Uintah::TypeDescription::Point) {
		  outfile << "VARIABLES =" << "\"X" << "\", " << "\"" << ccVariable << ".X" << "\" ";
	        }
	        if(subtype->getType() == Uintah::TypeDescription::Matrix3) {
		  outfile << "VARIABLES =" << "\"X" << "\", " << "\"" << ccVariable  << ".1.1\"";
	        }
	        outfile << endl;
	      }

	      //loop over the time
	      for(unsigned long t=time_step_lower;t<=time_step_upper;t=t+tskip){  //time loop: 6
		double time = times[t];
		cout << "time = " << time << endl;
	
		/////////////////////////////////////////////////////////////////
		// find index ranges for current grid level
		////////////////////////////////////////////////////////////////

		GridP grid = da->queryGrid(times[t]);
		for(int l=0;l<grid->numLevels();l++){  //level loop: 7
		  LevelP level = grid->getLevel(l);
		  cout << "\t    Level: " << level->getIndex() << ", id " << level->getID() << endl;

		  //		  int numNode,numPatch;
		  int numMatl;
		  int Imax,Jmax,Kmax,Imin,Jmin,Kmin;
		  int Irange, Jrange, Krange;
		  int indexI, indexJ, indexK;
		  IntVector lo,hi;
		  numMatl = 0;
		  Imax = 0;
		  Imin = 0;
		  Jmax = 0;
		  Jmin = 0;
		  Kmax = 0;
		  Kmin = 0;
		  Irange = 0;
		  Jrange = 0;
		  Krange = 0;
		  for(Level::const_patchIterator iter = level->patchesBegin();
		      iter != level->patchesEnd(); iter++){ // patch loop
		    const Patch* patch = *iter;
		    lo = patch->getLowIndex();
		    hi = patch->getHighIndex();
		    cout << "\t\tPatch: " << patch->getID() << " Over: " << lo << " to " << hi << endl;
		    int matlNum = da->queryNumMaterials(patch, time);
		    if(numMatl < matlNum) numMatl = matlNum;
		    if(Imax < hi.x()) Imax = hi.x();
		    if(Jmax < hi.y()) Jmax = hi.y();
		    if(Kmax < hi.z()) Kmax = hi.z();
		    if(Imin > lo.x()) Imin = lo.x();
		    if(Jmin > lo.y()) Jmin = lo.y();
		    if(Kmin > lo.z()) Kmin = lo.z();
		  } //patch loop
	    
		  Irange = Imax - Imin;             
		  Jrange = Jmax - Jmin;
		  Krange = Kmax - Kmin;

		  for(int matlsIndex = 0; matlsIndex < numMatl; matlsIndex++){ //matls loop: 8
		    //         write each Zone for diferent material at different time step for all patches for one of the levels
	          if((ccVariable != "delP_Dilatate" && ccVariable != "press_equil_CC" && ccVariable != "mach") ||
		    ((ccVariable == "delP_Dilatate" || ccVariable == "press_equil_CC") && matlsIndex == 0) ||
		     (ccVariable == "mach" && ( matlsIndex > 0))) {

		      if(i_xd == "i_3d"){ 
			outfile << "ZONE T =  " << "\"T:" << time << "," <<"M:" << matlsIndex << "," << "L:" << l << "," << "\"," 
				<< "N = " << Irange*Jrange*Krange << "," << "E = " << (Irange-1)*(Jrange-1)*(Krange-1) << "," 
				<< "F = " << "\"FEPOINT\"" << "," << "ET = " << "\"BRICK\"" << endl;
		      } else if(i_xd == "i_2d") {
			outfile << "ZONE T =  " <<"\"T:"  << time << "," <<"M:" << matlsIndex << ","  << "L:" << l << "\"," 
				<< "N = " << Irange*Jrange << "," << "E = " << (Irange-1)*(Jrange-1) << "," 
				<< "F = " << "\"FEPOINT\"" << "," << "ET = " << "\"QUADRILATERAL\"" << endl;
		      } else if(i_xd == "i_1d"){
			outfile << "ZONE T =  " <<"\"T:"  << time << "," <<"M:" << matlsIndex << ","  << "L:" << l << "\","
				<< "I = " << Irange << "," << "F = " << "\"POINT\"" << endl;
		      }

		      SCIRun::Array3<int> nodeIndex(Imax-Imin,Jmax-Jmin,Kmax-Kmin);
		      nodeIndex.initialize(0);

		      int totalNode = 0;
		      ////////////////////////////////////////////
		      // Write values of variable in current Zone
		      // /////////////////////////////////////////

		      for(Level::const_patchIterator iter = level->patchesBegin();
			  iter != level->patchesEnd(); iter++){ // patch loop: 9
			const Patch* patch = *iter;
			// get anchor, spacing for current level and patch
			Point start = level->getAnchor();
			Vector dx = patch->dCell();
			IntVector lo = patch->getLowIndex();
			IntVector hi = patch->getHighIndex();
			cout << "\t\tPatch: " << patch->getID() << " Over: " << lo << " to " << hi << endl;
			ConsecutiveRangeSet matls = da->queryMaterials(ccVariable, patch, time);
			for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
			    matlIter != matls.end(); matlIter++){ //material loop: 10
			  int matl = *matlIter;
			  if(matlsIndex == matl) { // if(matlsIndex == matl): 11
			    switch(subtype->getType()){ //switch to get data subtype: 12
			    case Uintah::TypeDescription::double_type:
			      {
				CCVariable<double> value;
				da->query(value, ccVariable, matl, patch, time);
				if(i_xd == "i_3d") {
				  for(indexK = lo.z(); indexK < hi.z(); ++indexK){
				    for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				      for(indexI = lo.x(); indexI < hi.x(); ++indexI){
					++totalNode;
					nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
					IntVector cellIndex(indexI, indexJ, indexK);
					outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
						<< start.y() + dx.y()*(indexJ + 1) << " "   
						<< start.z() + dx.z()*(indexK + 1) << " "  
						<< value[cellIndex] << endl;
				      }
				    }
				  } 
				} //(i_xd == "i_3d") 

				else if(i_xd == "i_2d") {
				  for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				    for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				      ++totalNode;
				      nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
				      IntVector cellIndex(indexI, indexJ, 0);
				      outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
					      << start.y() + dx.y()*(indexJ + 1) << " "   
					      << value[cellIndex] << endl;
				    }
				  }
				} //end of if(i_xd == "i_2d")

				else if(i_xd == "i_1d") {
				  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				    ++totalNode;
				    nodeIndex(indexI-Imin,0,0) = totalNode;
				    IntVector cellIndex(indexI, 0, 0);
				    outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
					    << value[cellIndex] << endl;
				  }
				}//end of if(i_xd == "i_1d") 
			      }
			    break;
			    case Uintah::TypeDescription::float_type:
			      {
				CCVariable<float> value;
				da->query(value, ccVariable, matl, patch, time);
				if(i_xd == "i_3d") {
				  for(indexK = lo.z(); indexK < hi.z(); ++indexK){
				    for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				      for(indexI = lo.x(); indexI < hi.x(); ++indexI){
					++totalNode;
					nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
					IntVector cellIndex(indexI, indexJ, indexK);
					outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
						<< start.y() + dx.y()*(indexJ + 1) << " "   
						<< start.z() + dx.z()*(indexK + 1) << " "  
						<< value[cellIndex] << endl;
				      }
				    }
				  } 
				} //(i_xd == "i_3d") 

				else if(i_xd == "i_2d") {
				  for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				    for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				      ++totalNode;
				      nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
				      IntVector cellIndex(indexI, indexJ, 0);
				      outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
					      << start.y() + dx.y()*(indexJ + 1) << " "   
					      << value[cellIndex] << endl;
				    }
				  }
				} //end of if(i_xd == "i_2d")

				else if(i_xd == "i_1d") {
				  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				    ++totalNode;
				    nodeIndex(indexI-Imin,0,0) = totalNode;
				    IntVector cellIndex(indexI, 0, 0);
				    outfile << start.x() + dx.x()*(indexI + 1) << " " //assume the begining index as [-1,-1,-1] 
					    << value[cellIndex] << endl;
				  }
				}//end of if(i_xd == "i_1d") 
			      }
			    break;
			    case Uintah::TypeDescription::Vector:
			      {
				CCVariable<Vector> value;
				da->query(value, ccVariable, matl, patch, time);
				if(i_xd == "i_3d") {
				  for(indexK = lo.z(); indexK < hi.z(); ++indexK){
				    for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				      for(indexI = lo.x(); indexI < hi.x(); ++indexI){
					IntVector cellIndex(indexI, indexJ, indexK);
					++totalNode;
					nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
					outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
						<< start.y() + dx.y()*(indexJ + 1) << " "
						<< start.z() + dx.z()*(indexK + 1) << " " 
						<< value[cellIndex].x() << " " << value[cellIndex].y() << " "
						<< value[cellIndex].z() << endl;  
				      }
				    }
				  }
				} // end of if(i_xd == "i_3d") 
				else if(i_xd == "i_2d"){
				  for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				    for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				      IntVector cellIndex(indexI, indexJ, 0);
				      ++totalNode;
				      nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
				      outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
					      << start.y() + dx.y()*(indexJ + 1) << " "
					      << value[cellIndex].x() << " " << value[cellIndex].y() << endl;
				    }
				  }
				} //end of if(i_xd == "i_2d")

				else if(i_xd == "i_1d") {
				  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				    IntVector cellIndex(indexI, 0, 0);
				    ++totalNode;
				    nodeIndex(indexI-Imin,0,0) = totalNode;
				    outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
					    << value[cellIndex].x() << endl;
				  }
				} //end of if(i_xd == "i_1d")
			      }
			    break;
			    case Uintah::TypeDescription::Point:
			      {
				CCVariable<Point> value;
				da->query(value, ccVariable, matl, patch, time);
				if(i_xd == "i_3d") {
				  for(indexK = lo.z(); indexK < hi.z(); ++indexK){
				    for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				      for(indexI = lo.x(); indexI < hi.x(); ++indexI){
					IntVector cellIndex(indexI, indexJ, indexK);
					++totalNode;
					nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
					outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
						<< start.y() + dx.y()*(indexJ + 1) << " "
						<< start.z() + dx.z()*(indexK + 1) << " " 
						<< value[cellIndex].x() << " " << value[cellIndex].y() << " "
						<< value[cellIndex].z() << endl;  
				      }
				    }
				  }
				} // end of if(i_xd == "i_3d") 
			   
				else if(i_xd == "i_2d") {
				  for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				    for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				      IntVector cellIndex(indexI, indexJ, 0);
				      ++totalNode;
				      nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
				      outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
					      << start.y() + dx.y()*(indexJ + 1) << " "
					      << value[cellIndex].x() << " " << value[cellIndex].y() << endl;
				    }
				  }
				} //end of if(i_xd == "i_2d")

				else if(i_xd == "i_1d") {
				  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				    IntVector cellIndex(indexI, 0, 0);
				    ++totalNode;
				    nodeIndex(indexI-Imin,0,0) = totalNode;
				    outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
					    << value[cellIndex].x() << endl;
				  }
				} //end of if(i_xd == "i_1d")
			      }
			    break;

			    case Uintah::TypeDescription::Matrix3:
			      {
				CCVariable<Matrix3> value;
				da->query(value, ccVariable, matl, patch, time);
				if(i_xd == "i_3d") {
				  for(indexK = lo.z(); indexK < hi.z(); ++indexK){
				    for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				      for(indexI = lo.x(); indexI < hi.x(); ++indexI){
					++totalNode;
					nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) = totalNode;
					IntVector cellIndex(indexI, indexJ, indexK);
					outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
						<< start.y() + dx.y()*(indexJ + 1) << " " 
						<< start.z() + dx.z()*(indexK + 1) << " "   
						<< (value[cellIndex])(0,0) << " " << (value[cellIndex])(0,1) << " " 
						<< (value[cellIndex])(0,2) << " " 
						<< (value[cellIndex])(1,0) << " " << (value[cellIndex])(1,1) << " "  
						<< (value[cellIndex])(1,2) << " " 
						<< (value[cellIndex])(2,0) << " " << (value[cellIndex])(2,1) << " " 
						<< (value[cellIndex])(2,2) << endl;  
				      }
				    }
				  }
				}//end of if(i_xd == "i_3d")
		      
				else if(i_xd == "i_2d"){
				  for(indexJ = lo.y(); indexJ < hi.y(); ++indexJ){
				    for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				      ++totalNode;
				      nodeIndex(indexI-Imin,indexJ-Jmin,0) = totalNode;
				      IntVector cellIndex(indexI, indexJ, 0);
				      outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
					      << start.y() + dx.y()*(indexJ + 1) << " "  
					      << (value[cellIndex])(0,0) << " " << (value[cellIndex])(0,1) << " "
					      << (value[cellIndex])(1,0) << " " << (value[cellIndex])(1,1) << " "
					      << endl;
				    }
				  }
				}//end of if(i_xd == "i_2d")

				else if(i_xd == "i_1d"){
				  for(indexI = lo.x(); indexI < hi.x(); ++indexI){
				    ++totalNode;
				    nodeIndex(indexI-Imin,0,0) = totalNode;
				    IntVector cellIndex(indexI, 0, 0);
				    outfile << start.x() + dx.x()*(indexI + 1) << " "  //assume the begining index is [-1,-1,-1]
					    << (value[cellIndex])(0,0) << endl; 
				  }
				}//end of if(i_xd == "i_1d") 
			      }
			    break;
			    default:
			      cerr << "CC Variable of unknown type: " << subtype->getName() << endl;
			      break;
			    } //end of switch (subtype->getType()): 12
			  } //end of if(matlsIndex == matl): 11
			} //end of matls loop: 10
		      } // end of loop over patches: 9
	    
		      //////////////////////////////////////////////////////////////////////////////////////////////
		      // Write connectivity list in current Zone
		      /////////////////////////////////////////////////////////////////////////////////////////////
		      if(i_xd == "i_3d"){
			for(indexK = Kmin; indexK < Kmax-1; ++indexK){
			  for(indexJ = Jmin; indexJ < Jmax-1; ++indexJ){
                            for(indexI = Imin; indexI < Imax-1; ++indexI){
			      outfile << nodeIndex(indexI-Imin,indexJ-Jmin,indexK-Kmin) << " "  
				      << nodeIndex(indexI+1-Imin,indexJ-Jmin,indexK-Kmin) << " "  
				      << nodeIndex(indexI+1-Imin,indexJ+1-Jmin,indexK-Kmin) << " "  
				      << nodeIndex(indexI-Imin,indexJ+1-Jmin,indexK-Kmin) << " "  
				      << nodeIndex(indexI-Imin,indexJ-Jmin,indexK+1-Kmin) << " "  
				      << nodeIndex(indexI+1-Imin,indexJ-Jmin,indexK+1-Kmin) << " "  
				      << nodeIndex(indexI+1-Imin,indexJ+1-Jmin,indexK-Kmin) << " "  
				      << nodeIndex(indexI-Imin,indexJ+1-Jmin,indexK+1-Kmin) << endl;
			    } //end of loop over indexI
			  } //end of loop over indexJ
			} //end of loop over indexK
		      }//end of if(i_xd == "i_3d") 
			
		      else if(i_xd == "i_2d"){
			for(indexJ = Jmin; indexJ < Jmax-1; ++indexJ){
			  for(indexI = Imin; indexI < Imax-1; ++indexI){
			    outfile << nodeIndex(indexI-Imin,indexJ-Jmin,0) << " "  
				    << nodeIndex(indexI+1-Imin,indexJ-Jmin,0) << " "  
				    << nodeIndex(indexI+1-Imin,indexJ+1-Jmin,0) << " "  
				    << nodeIndex(indexI-Imin,indexJ+1-Jmin,0) << " "  
				    << endl;
			  } //end of loop over indexI
			} //end of loop over indexJ
		      } //end of if(i_xd == "i_2d") 
		    } // end of pressure ccVariable if: 8'9
		  } // end of loop over matlsIndex: 8
		} // end of loop over levels: 7
	      } // end of loop over times: 6
	    }//end of CCVariable case: 5
	  break;
	  default:
	    // for other variables in the future
	    break;
	  } // end switch( td->getType() ): 4
	} // end of if block (do_all_ccvars || ...): 3
      } // end of loop over variables: 2
    } //end of if (do_tecplot): 1

    //______________________________________________________________________
    //              V A R S U M M A R Y   O P T I O N
    if(do_varsummary){
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
      for(int i=0;i<(int)index.size();i++)
	cout << index[i] << ": " << times[i] << endl;
      
       findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
      
      for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
	double time = times[t];
	cout << "time = " << time << endl;
	GridP grid = da->queryGrid(time);
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];
	  const Uintah::TypeDescription* subtype = td->getSubType();
	  cout << "\tVariable: " << var << ", type " << td->getName() << endl;
	  for(int l=0;l<grid->numLevels();l++){
	    LevelP level = grid->getLevel(l);
	    cout << "\t    Level: " << level->getIndex() << ", id " << level->getID() << endl;
	    for(Level::const_patchIterator iter = level->patchesBegin();
		iter != level->patchesEnd(); iter++){
	      const Patch* patch = *iter;
	      cout << "\t\tPatch: " << patch->getID() << endl;
	      ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	      // loop over materials
	      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		  matlIter != matls.end(); matlIter++){
		int matl = *matlIter;
		cout << "\t\t\tMaterial: " << matl << endl;
		switch(td->getType()){
		  //__________________________________
		  //   P A R T I C L E   V A R I A B L E
		case Uintah::TypeDescription::ParticleVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      ParticleVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			double min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++];
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      ParticleVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			float min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++];
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::int_type:
		    {
		      ParticleVariable<int> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			int min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++];
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      ParticleVariable<Point> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			Point min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter];
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      ParticleVariable<Vector> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			double min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++].length2();
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter].length2());
			  max=Max(max, value[*iter].length2());
			}
			cout << "\t\t\t\tmin magnitude: " << sqrt(min) << endl;
			cout << "\t\t\t\tmax magnitude: " << sqrt(max) << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Matrix3:
		    {
		      ParticleVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			double min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++].Norm();
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter].Norm());
			  max=Max(max, value[*iter].Norm());
			}
			cout << "\t\t\t\tmin Norm: " << min << endl;
			cout << "\t\t\t\tmax Norm: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::long64_type:
		    {
		      ParticleVariable<long64> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cout << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			long64 min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++];
			for(;iter != pset->end(); iter++){
			  min=std::min(min, value[*iter]);
			  max=std::max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin : " << min << endl;
			cout << "\t\t\t\tmax : " << max << endl;
		      }
		    }
		  break;
		  default:
		    cerr << "Particle Variable of unknown type: " << subtype->getName() << endl;
		    break;
		  }
		  break;
		  //__________________________________  
		  //  N C   V A R I A B L E S           
		case Uintah::TypeDescription::NCVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      NCVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi = value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      NCVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi = value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			float min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      NCVariable<Point> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			Point min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      NCVariable<Vector> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi = value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter].length2();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].length2());
			  max=Max(max, value[*iter].length2());
			}
			cout << "\n";
			cout << "\t\t\t\tmin magnitude: " << sqrt(min) << endl;
			cout << "\t\t\t\tmax magnitude: " << sqrt(max) << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Matrix3:
		    {
		      NCVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter].Norm();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].Norm());
			  max=Max(max, value[*iter].Norm());
			}
			cout << "\t\t\t\tmin Norm: " << min << endl;
			cout << "\t\t\t\tmax Norm: " << max << endl;
		      }
		    }
		  break;
		  default:
		    cerr << "NC Variable of unknown type: " << subtype->getName() << endl;
		    break;
		  }
		  break;
		  //__________________________________
		  //   C C   V A R I A B L E S
		case Uintah::TypeDescription::CCVariable:
		  switch(subtype->getType()){
                  case Uintah::TypeDescription::int_type:
		    {
		      CCVariable<int> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			int min, max;
			IntVector c_min, c_max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  int val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::double_type:
		    {
		      CCVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			IntVector c_min, c_max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  double val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      CCVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			float min, max;
			IntVector c_min, c_max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  float val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      CCVariable<Point> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			Point min, max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << endl;
			cout << "\t\t\t\tmax value: " << max << endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      CCVariable<Vector> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			IntVector c_min, c_max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter].length2();
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  double val = value[*iter].length2();
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin magnitude: " << sqrt(min) << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax magnitude: " << sqrt(max) << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Matrix3:
		    {
		      CCVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter].Norm();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].Norm());
			  max=Max(max, value[*iter].Norm());
			}
			cout << "\t\t\t\tmin Norm: " << min << endl;
			cout << "\t\t\t\tmax Norm: " << max << endl;
		      }
		    }
		  break;
		  default:
		    cerr << "CC Variable of unknown type: " << subtype->getName() << endl;
		    break;
		  }
		  break;
		  //__________________________________
		  //   S F C X   V A R I A B L E S
		case Uintah::TypeDescription::SFCXVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      SFCXVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
                    
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			IntVector c_min, c_max;
			
			CellIterator iter=patch->getSFCXIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  double val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      SFCXVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
                    
		      if(dx.x() && dx.y() && dx.z()){
			float min, max;
			IntVector c_min, c_max;
			
			CellIterator iter=patch->getSFCXIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  float val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  default:
		    cerr << "SCFXVariable  of unknown type: " << subtype->getType() << endl;
		    break;
		  }
		  break;
		  //__________________________________
		  //   S F C Y  V A R I A B L E S
		case Uintah::TypeDescription::SFCYVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      SFCYVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
                    
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			IntVector c_min, c_max;
			
			CellIterator iter=patch->getSFCYIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  double val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      SFCYVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
                    
		      if(dx.x() && dx.y() && dx.z()){
			float min, max;
			IntVector c_min, c_max;
			
			CellIterator iter=patch->getSFCYIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  float val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  default:
		    cerr << "SCFYVariable  of unknown type: " << subtype->getType() << endl;
		    break;
		  }
		  break;
		  //__________________________________
		  //   S F C Z   V A R I A B L E S
		case Uintah::TypeDescription::SFCZVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      SFCZVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
                    
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			IntVector c_min, c_max;
			
			CellIterator iter=patch->getSFCZIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  double val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      SFCZVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      IntVector lo = value.getLowIndex();
		      IntVector hi= value.getHighIndex() - IntVector(1,1,1);
		      cout << "\t\t\t\t" << td->getName() << " over " << lo << " to " << hi << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
                    
		      if(dx.x() && dx.y() && dx.z()){
			float min, max;
			IntVector c_min, c_max;
			
			CellIterator iter=patch->getSFCZIterator();
			min=max=value[*iter];
			c_min = c_max = *iter;
			// No need to do a comparison on the initial cell
			iter++;
			for(;!iter.done(); iter++){
			  float val = value[*iter];
			  if (val < min) {
			    min = val;
			    c_min = *iter;
			  }
			  if (val > max ) {
			    max = val;
			    c_max = *iter;
			  }
			}
			cout << "\t\t\t\tmin value: " << min << "\t\t"<< c_min <<endl;
			cout << "\t\t\t\tmax value: " << max << "\t\t"<< c_max <<endl;
		      }
		    }
		  break;
		  default:
		    cerr << "SCFZVariable  of unknown type: " << subtype->getType() << endl;
		    break;
		  }
		  break;
		  //__________________________________
		  //  BULLET PROOFING
		default:
		  cerr << "Variable of unknown type: " << td->getName() << endl;
		  break;
		}
	      }
	    }
	  }
	}
      }
    }

    //______________________________________________________________________
    //              J I M 1   O P T I O N
    // This currently pulls out particle position, velocity and ID
    // and prints that on one line for each particle.  This is useful
    // for postprocessing of particle data for particles which move
    // across patches.
    if(do_jim1){
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
      for(int i=0;i<(int)index.size();i++)
	cout << index[i] << ": " << times[i] << endl;
      
       findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
      
      for(unsigned long t=time_step_lower;t<=time_step_upper;t+=time_step_inc){
	double time = times[t];
	cout << "time = " << time << endl;
	GridP grid = da->queryGrid(time);
          ostringstream fnum;
          string filename;
          fnum << setw(4) << setfill('0') << t/time_step_inc;
          string partroot("partout");
          filename = partroot+ fnum.str();
          ofstream partfile(filename.c_str());

	  for(int l=0;l<grid->numLevels();l++){
	    LevelP level = grid->getLevel(l);
	    cout << "Level: " <<  endl;
	    for(Level::const_patchIterator iter = level->patchesBegin();
		iter != level->patchesEnd(); iter++){
	      const Patch* patch = *iter;
		int matl = 0;
		  //__________________________________
		  //   P A R T I C L E   V A R I A B L E
		  ParticleVariable<long64> value_pID;
		  ParticleVariable<Point> value_pos;
		  ParticleVariable<Vector> value_vel;
                  da->query(value_pID, "p.particleID", matl, patch, time);
                  da->query(value_pos, "p.x",          matl, patch, time);
                  da->query(value_vel, "p.velocity",   matl, patch, time);
                  ParticleSubset* pset = value_pos.getParticleSubset();
                  if(pset->numParticles() > 0){
                     ParticleSubset::iterator iter = pset->begin();
                     for(;iter != pset->end(); iter++){
                         partfile << value_pos[*iter].x() << " " <<
                                     value_pos[*iter].y() << " " <<
                                     value_pos[*iter].z() << " "; 
                         partfile << value_vel[*iter].x() << " " <<
                                     value_vel[*iter].y() << " " <<
                                     value_vel[*iter].z() << " "; 
			 partfile << value_pID[*iter] <<  endl;
                     } // for
		  }  //if
	    }  // for patches
	  }   // for levels
      }
    }
    //______________________________________________________________________
    //    DO_ASCI
    if (do_asci){
      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      ASSERTEQ(vars.size(), types.size());
      int freq = 1; int ts=1;
      
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      if (index.size() == 1)
      	cout << "There is only 1 timestep:\n";
      else
	cout << "There are " << index.size() << " timesteps:\n";
      
      findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
      
      // Loop over time
      for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
	double time = times[t];
      	int partnum = 1;
	int num_of_particles = 0;
	cout << "timestep " << ts << " inprogress... ";
	
  	if (( ts % freq) == 0) {
   		
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
	
	  GridP grid = da->queryGrid(time);
	  int l=0;
	  LevelP level = grid->getLevel(l);
	  Level::const_patchIterator iter = level->patchesBegin();
	  const Patch* patch = *iter;
		
		
	  // for loop over variables for name printing
	  for(unsigned int v=0;v<vars.size();v++){
	    std::string var = vars[v];
	       
	    ConsecutiveRangeSet matls= da->queryMaterials(var, patch, time);
	    // loop over materials
	    for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		matlIter != matls.end(); matlIter++){
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
	  grid = da->queryGrid(time);
	  level = grid->getLevel(l);
	  iter = level->patchesBegin();
	  patch = *iter;
	
	  // loop over variables for printing values
	  for(unsigned int v=0;v<vars.size();v++){
	    std::string var = vars[v];
		
	    ConsecutiveRangeSet matls=da->queryMaterials(var, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
		    da->query(value, var, matl, patch, time);
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
    } //end of do_asci		
    //______________________________________________________________________
    //	       DO CELL STRESSES	
    if (do_cell_stresses){
      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      ASSERTEQ(vars.size(), types.size());
      
      cout << "There are " << vars.size() << " variables:\n";
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      
      cout << "There are " << index.size() << " timesteps:\n";
      
      findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
      
      // obtain the desired timesteps
      unsigned long t = 0, start_time, stop_time;

      cout << "Time Step       Value\n";
      
      for(t = time_step_lower; t <= time_step_upper; t++){
	double time = times[t];
	cout << "    " << t + 1 << "        "  << time << endl;
      }
      cout << endl;
      if (t != (time_step_lower +1)){
	cout << "Enter start time-step (1 - " << t << "): ";
	cin >> start_time;
	start_time--;
	cout << "Enter stop  time-step (1 - " << t << "): ";
	cin >> stop_time;
	stop_time--;
      }
      else 
      	if(t == (time_step_lower + 1)){
	  start_time = t-1;
	  stop_time  = t-1;
	}
      // end of timestep acquisition
      
      for(t=start_time;t<=stop_time;t++){
	
	double time = times[t];
	cout << "time = " << time << endl;
	GridP grid = da->queryGrid(time);
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  
	  // only dumps out data if it is variable g.stressFS
	  if (var == "g.stressFS"){
	    const Uintah::TypeDescription* td = types[v];
	    const Uintah::TypeDescription* subtype = td->getSubType();
	    cout << "\tVariable: " << var << ", type " << td->getName() << endl;
	    for(int l=0;l<grid->numLevels();l++){
	      LevelP level = grid->getLevel(l);
	      for(Level::const_patchIterator iter = level->patchesBegin();
		  iter != level->patchesEnd(); iter++){
		const Patch* patch = *iter;
		cout << "\t\tPatch: " << patch->getID() << endl;
                ConsecutiveRangeSet matls =
		  da->queryMaterials(var, patch, time);
	        // loop over materials
	        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		    matlIter != matls.end(); matlIter++){
		  int matl = *matlIter;
		  
		  // dumps header and variable info to file
		  ostringstream fnum, pnum, matnum; 
		  string filename;
		  unsigned long timestepnum=t+1;
		  fnum << setw(4) << setfill('0') << timestepnum;
                  pnum << setw(4) << setfill('0') << patch->getID();
		  matnum << setw(4) << setfill('0') << matl;
		  string partroot("stress.t");
                  string partextp(".p"); 
		  string partextm(".m");
		  filename = partroot+fnum.str()+partextp+pnum.str()+partextm+matnum.str();
		  ofstream partfile(filename.c_str());
		  partfile << "# x, y, z, st11, st12, st13, st21, st22, st23, st31, st32, st33" << endl;
		  
		  cout << "\t\t\tMaterial: " << matl << endl;
		  switch(td->getType()){
		  case Uintah::TypeDescription::NCVariable:
		    switch(subtype->getType()){
		    case Uintah::TypeDescription::Matrix3:{
		      NCVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex()
			   << " to " << value.getHighIndex() << endl;
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			NodeIterator iter = patch->getNodeIterator();
			for(;!iter.done(); iter++){
			  partfile << (*iter).x() << " " << (*iter).y() << " " << (*iter).z()
				   << " " << (value[*iter])(0,0) << " " << (value[*iter])(0,1) << " " 
				   << (value[*iter])(0,2) << " " << (value[*iter])(1,0) << " "
				   << (value[*iter])(1,1) << " " << (value[*iter])(1,2) << " "
				   << (value[*iter])(2,0) << " " << (value[*iter])(2,1) << " "
                                   << (value[*iter])(2,2) << endl;
			}
		      }
		    }
		      break;
		    default:
		      cerr << "No Matrix3 Subclass avaliable." << subtype->getType() << endl;
		      break;
		    }
		    break;
		  default:
		    cerr << "No NC Variables avaliable." << td->getType() << endl;
		    break;
		  }
		}
	      }
	    }
	  }
	  else
	    cout << "No g.stressFS variables avaliable at time " << t << "." << endl;
	}
	if (start_time == stop_time)
	  t++;   
      }
    }
    
    //______________________________________________________________________
    //              DO RTDATA
    if (do_rtdata) {
      // Create a directory if it's not already there.
      // The exception occurs when the directory is already there
      // and the Dir.create fails.  This exception is ignored. 
      if(raydatadir != "") {
	Dir rayDir;
	try {
	  rayDir.create(raydatadir);
	}
	catch (Exception& e) {
	  cerr << "Caught exception: " << e.message() << endl;
	}
      }

      // set up the file that contains a list of all the files
      FILE* filelist;
      string filelistname = raydatadir + string("/") + string("timelist");
      filelist = fopen(filelistname.c_str(),"w");
      if (!filelist) {
	cerr << "Can't open output file " << filelistname << endl;
	abort();
      }

      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      ASSERTEQ(vars.size(), types.size());
      cout << "There are " << vars.size() << " variables:\n";
      
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";

      std::string time_file;
      std::string variable_file;
      std::string patchID_file;
      std::string materialType_file;
      
      findTimestep_loopLimits(tslow_set, tsup_set,times, time_step_lower, time_step_upper);
      
      // for all timesteps
      for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
	double time = times[t];
	ostringstream tempstr_time;
	tempstr_time << setprecision(17) << time;
	time_file = replaceChar(string(tempstr_time.str()),'.','_');
	GridP grid = da->queryGrid(time);
	fprintf(filelist,"<TIMESTEP>\n");
	if(do_verbose)
	  cout << "time = " << time << endl;
	// Create a directory if it's not already there.
	// The exception occurs when the directory is already there
	// and the Dir.create fails.  This exception is ignored. 
	Dir rayDir;
	try {
	  rayDir.create(raydatadir + string("/TS_") + time_file);
	}
	catch (Exception& e) {
	  cerr << "Caught directory making exception: " << e.message() << endl;
	}
	// for each level in the grid
	for(int l=0;l<grid->numLevels();l++){
	  LevelP level = grid->getLevel(l);
	  
	  // for each patch in the level
	  for(Level::const_patchIterator iter = level->patchesBegin();
	      iter != level->patchesEnd(); iter++){
	    const Patch* patch = *iter;
	    ostringstream tempstr_patch;
	    tempstr_patch << patch->getID();
	    patchID_file = tempstr_patch.str();
	    fprintf(filelist,"<PATCH>\n");

	    vector<MaterialData> material_data_list; 
	    	    
	    // for all vars in one timestep in one patch
	    for(int v=0;v<(int)vars.size();v++){
	      std::string var = vars[v];
	      //cerr << "***********Found variable " << var << "*********\n";
	      variable_file = replaceChar(var,'.','_');
	      const Uintah::TypeDescription* td = types[v];
	      const Uintah::TypeDescription* subtype = td->getSubType();

	      ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	      // loop over materials
	      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		  matlIter != matls.end(); matlIter++){
		int matl = *matlIter;
		ostringstream tempstr_matl;
		tempstr_matl << matl;
		materialType_file = tempstr_matl.str();

		MaterialData material_data;

		if (matl <(int) material_data_list.size())
		  material_data = material_data_list[matl];
		
	        switch(td->getType()){
	        case Uintah::TypeDescription::ParticleVariable:
		  if (do_PTvar) {
		    switch(subtype->getType()){
		    case Uintah::TypeDescription::double_type:
		      {
			ParticleVariable<double> value;
			da->query(value, var, matl, patch, time);
			material_data.pv_double_list.push_back(value);
		      }
		    break;
		    case Uintah::TypeDescription::float_type:
		      {
			ParticleVariable<float> value;
			da->query(value, var, matl, patch, time);
			material_data.pv_float_list.push_back(value);
		      }
		    break;
		    case Uintah::TypeDescription::Point:
		      {
			ParticleVariable<Point> value;
			da->query(value, var, matl, patch, time);

			//cout << __LINE__ << ":var = ("<<var<<") for material index "<<matl<<"\n";
			if (var == "p.x") {
			  //cout << "Found p.x!\n";
			  material_data.p_x.copyPointer(value);
			} else {
			  material_data.pv_point_list.push_back(value);
			}
		      }
		    break;
		    case Uintah::TypeDescription::Vector:
		      {
			ParticleVariable<Vector> value;
			da->query(value, var, matl, patch, time);
			material_data.pv_vector_list.push_back(value);
		      }
		    break;
		    case Uintah::TypeDescription::Matrix3:
		      {
			ParticleVariable<Matrix3> value;
			da->query(value, var, matl, patch, time);
			material_data.pv_matrix3_list.push_back(value);
		      }
		    break;
		    default:
		      cerr << __LINE__ << ":Particle Variable of unknown type: " << subtype->getName() << endl;
		      break;
		    }
		    break;
		  }
		case Uintah::TypeDescription::NCVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      if (do_NCvar_double) {
			// setup output files
			string raydatafile = makeFileName(raydatadir,variable_file,time_file,patchID_file,materialType_file);			
			FILE* datafile;
			FILE* headerfile;
			if (!setupOutFiles(&datafile,&headerfile,raydatafile,string("hdr")))
			  abort();

			// addfile to filelist
			fprintf(filelist,"%s\n",raydatafile.c_str());
			// get the data and write it out
			double min = 0.0, max = 0.0;
			NCVariable<double> value;
			da->query(value, var, matl, patch, time);
			IntVector dim(value.getHighIndex()-value.getLowIndex());
			if(dim.x() && dim.y() && dim.z()){
			  NodeIterator iter = patch->getNodeIterator();
			  min=max=value[*iter];
			  for(;!iter.done(); iter++){
			    min=Min(min, value[*iter]);
			    max=Max(max, value[*iter]);
			    float temp_value = (float)value[*iter];
			    fwrite(&temp_value, sizeof(float), 1, datafile);
			  }	  
			}
			
			Point b_min = patch->getBox().lower();
			Point b_max = patch->getBox().upper();
			
			// write the header file
			fprintf(headerfile, "%d %d %d\n",dim.x(), dim.y(), dim.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_min.x(),(float)b_min.y(),(float)b_min.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_max.x(),(float)b_max.y(),(float)b_max.z());
			fprintf(headerfile, "%f %f\n",(float)min,(float)max);

			fclose(datafile);
			fclose(headerfile);
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      if (do_NCvar_float) {
			// setup output files
			string raydatafile = makeFileName(raydatadir,variable_file,time_file,patchID_file,materialType_file);			
			FILE* datafile;
			FILE* headerfile;
			if (!setupOutFiles(&datafile,&headerfile,raydatafile,string("hdr")))
			  abort();

			// addfile to filelist
			fprintf(filelist,"%s\n",raydatafile.c_str());
			// get the data and write it out
			float min = 0.0, max = 0.0;
			NCVariable<float> value;
			da->query(value, var, matl, patch, time);
			IntVector dim(value.getHighIndex()-value.getLowIndex());
			if(dim.x() && dim.y() && dim.z()){
			  NodeIterator iter = patch->getNodeIterator();
			  min=max=value[*iter];
			  for(;!iter.done(); iter++){
			    min=Min(min, value[*iter]);
			    max=Max(max, value[*iter]);
			    float temp_value = value[*iter];
			    fwrite(&temp_value, sizeof(float), 1, datafile);
			  }	  
			}
			
			Point b_min = patch->getBox().lower();
			Point b_max = patch->getBox().upper();
			
			// write the header file
			fprintf(headerfile, "%d %d %d\n",dim.x(), dim.y(), dim.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_min.x(),(float)b_min.y(),(float)b_min.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_max.x(),(float)b_max.y(),(float)b_max.z());
			fprintf(headerfile, "%f %f\n",(float)min,(float)max);

			fclose(datafile);
			fclose(headerfile);
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      if (do_NCvar_point) {
			// not implemented at this time
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      if (do_NCvar_vector) {
			// not implemented at this time
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Matrix3:
		    {
		      if (do_NCvar_matrix3) {
			// not implemented at this time
		      }
		    }
		  break;
		  default:
		    cerr << "NC variable of unknown type: " << subtype->getName() << endl;
		    break;
		  }
		  break;
		case Uintah::TypeDescription::CCVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      if (do_CCvar_double) {
			// setup output files
			string raydatafile = makeFileName(raydatadir,variable_file,time_file,patchID_file,materialType_file);			
			FILE* datafile;
			FILE* headerfile;
			if (!setupOutFiles(&datafile,&headerfile,raydatafile,string("hdr")))
			  abort();

			// addfile to filelist
			fprintf(filelist,"%s\n",raydatafile.c_str());
			// get the data and write it out
			double min = 0.0, max = 0.0;
			CCVariable<double> value;
			da->query(value, var, matl, patch, time);
			IntVector dim(value.getHighIndex()-value.getLowIndex());
			if(dim.x() && dim.y() && dim.z()){
			  NodeIterator iter = patch->getNodeIterator();
			  min=max=value[*iter];
			  for(;!iter.done(); iter++){
			    min=Min(min, value[*iter]);
			    max=Max(max, value[*iter]);
			    float temp_value = (float)value[*iter];
			    fwrite(&temp_value, sizeof(float), 1, datafile);
			  }	  
			}
			
			Point b_min = patch->getBox().lower();
			Point b_max = patch->getBox().upper();
			
			// write the header file
			fprintf(headerfile, "%d %d %d\n",dim.x(), dim.y(), dim.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_min.x(),(float)b_min.y(),(float)b_min.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_max.x(),(float)b_max.y(),(float)b_max.z());
			fprintf(headerfile, "%f %f\n",(float)min,(float)max);

			fclose(datafile);
			fclose(headerfile);
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      if (do_CCvar_float) {
			// setup output files
			string raydatafile = makeFileName(raydatadir,variable_file,time_file,patchID_file,materialType_file);			
			FILE* datafile;
			FILE* headerfile;
			if (!setupOutFiles(&datafile,&headerfile,raydatafile,string("hdr")))
			  abort();

			// addfile to filelist
			fprintf(filelist,"%s\n",raydatafile.c_str());
			// get the data and write it out
			float min = 0.0, max = 0.0;
			CCVariable<float> value;
			da->query(value, var, matl, patch, time);
			IntVector dim(value.getHighIndex()-value.getLowIndex());
			if(dim.x() && dim.y() && dim.z()){
			  NodeIterator iter = patch->getNodeIterator();
			  min=max=value[*iter];
			  for(;!iter.done(); iter++){
			    min=Min(min, value[*iter]);
			    max=Max(max, value[*iter]);
			    float temp_value = value[*iter];
			    fwrite(&temp_value, sizeof(float), 1, datafile);
			  }	  
			}
			
			Point b_min = patch->getBox().lower();
			Point b_max = patch->getBox().upper();
			
			// write the header file
			fprintf(headerfile, "%d %d %d\n",dim.x(), dim.y(), dim.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_min.x(),(float)b_min.y(),(float)b_min.z());
			fprintf(headerfile, "%f %f %f\n",(float)b_max.x(),(float)b_max.y(),(float)b_max.z());
			fprintf(headerfile, "%f %f\n",(float)min,(float)max);

			fclose(datafile);
			fclose(headerfile);
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      if (do_CCvar_point) {
			// not implemented at this time
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      if (do_CCvar_vector) {
			// not implemented at this time
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Matrix3:
		    {
		      if (do_CCvar_matrix3) {
			// not implemented at this time
		      }
		    }
		  break;
		  default:
		    cerr << "CC variable of unknown type: " << subtype->getName() << endl;
		    break;
		  }
		  break;
		default:
		  cerr << "Variable of unknown type: " << td->getName() << endl;
		  break;
		} // end switch(td->getType())
		if (matl < (int)material_data_list.size())
		  material_data_list[matl] = material_data;
		else
		  material_data_list.push_back(material_data);
	      } // end matl
	      
	    } // end vars
	    // after all the variable data has been collected write it out
	    if (do_PTvar) {
	      FILE* datafile;
	      FILE* headerfile;
	      //--------------------------------------------------
	      // set up the first min/max
	      Point min, max;
	      vector<double> d_min,d_max,f_min,f_max,v_min,v_max,m_min,m_max;
	      bool data_found = false;
	      int total_particles = 0;
	      
	      // loops until a non empty material_data set has been
	      // found and inialized the mins and maxes
	      for(int m = 0; m <(int) material_data_list.size(); m++) {
		// determine the min and max
		MaterialData md = material_data_list[m];
		//cerr << "First md = " << m << endl;
		ParticleSubset* pset = md.p_x.getParticleSubset();
		if (!pset) {
		  cerr << __LINE__ << ":No particle location variable found for material index "<<m<<"\n";
		  continue;
		  abort();
		}
		int numParticles = pset->numParticles();
		if(numParticles > 0){
		  ParticleSubset::iterator iter = pset->begin();

		  // setup min/max for p.x
		  min=max=md.p_x[*iter];
		  // setup min/max for all others
		  if (do_PTvar_all) {
		    for(int i = 0; i <(int) md.pv_double_list.size(); i++) {
		      d_min.push_back(md.pv_double_list[i][*iter]);
		      d_max.push_back(md.pv_double_list[i][*iter]);
		    }
		    for(int i = 0; i <(int) md.pv_float_list.size(); i++) {
		      f_min.push_back(md.pv_float_list[i][*iter]);
		      f_max.push_back(md.pv_float_list[i][*iter]);
		    }
		    for(int i = 0; i < (int)md.pv_vector_list.size(); i++) {
		      v_min.push_back(md.pv_vector_list[i][*iter].length());
		      v_max.push_back(md.pv_vector_list[i][*iter].length());
		    }
		    for(int i = 0; i < (int)md.pv_matrix3_list.size(); i++) {
		      m_min.push_back(md.pv_matrix3_list[i][*iter].Norm());
		      m_max.push_back(md.pv_matrix3_list[i][*iter].Norm());
		    }
		  }
		  // initialized mins/maxes
		  data_found = true;
		  // setup output files
		  string raydatafile = makeFileName(raydatadir,string(""),time_file,patchID_file,string(""));
		  if (!setupOutFiles(&datafile,&headerfile,raydatafile,string("meta")))
		    abort();
		  // addfile to filelist
		  fprintf(filelist,"%s\n",raydatafile.c_str());
		  
		  break;
		}
		
	      }

	      //--------------------------------------------------
	      // extract data and write it to a file MaterialData at a time

	      if (do_verbose)
		cerr << "---Extracting data and writing it out  ";
	      for(int m = 0; m <(int) material_data_list.size(); m++) {
		MaterialData md = material_data_list[m];
		ParticleSubset* pset = md.p_x.getParticleSubset();
		// a little redundant, but may not have been cought
		// by the previous section
		if (!pset) {
		  cerr << __LINE__ << ":No particle location variable found\n";
		  continue;
		  abort();
		}
		
		int numParticles = pset->numParticles();
		total_particles+= numParticles;
		if(numParticles > 0){
		  ParticleSubset::iterator iter = pset->begin();
		  for(;iter != pset->end(); iter++){
		    // p_x
		    min=Min(min, md.p_x[*iter]);
		    max=Max(max, md.p_x[*iter]);
		    float temp_value = (float)(md.p_x[*iter]).x();
		    fwrite(&temp_value, sizeof(float), 1, datafile);
		    temp_value = (float)(md.p_x[*iter]).y();
		    fwrite(&temp_value, sizeof(float), 1, datafile);
		    temp_value = (float)(md.p_x[*iter]).z();
		    fwrite(&temp_value, sizeof(float), 1, datafile);
		    if (do_PTvar_all) {
		      // double data
		      for(int i = 0; i <(int) md.pv_double_list.size(); i++) {
			double value = md.pv_double_list[i][*iter];
			d_min[i]=Min(d_min[i],value);
			d_max[i]=Max(d_max[i],value);
			temp_value = (float)value;
			fwrite(&temp_value, sizeof(float), 1, datafile);
		      }
		      // float data
		      for(int i = 0; i <(int) md.pv_float_list.size(); i++) {
			float value = md.pv_float_list[i][*iter];
			f_min[i]=Min(f_min[i],(double)value);
			f_max[i]=Max(f_max[i],(double)value);
			temp_value = value;
			fwrite(&temp_value, sizeof(float), 1, datafile);
		      }
		      // vector data
		      for(int i = 0; i < (int)md.pv_vector_list.size(); i++) {
			double value = md.pv_vector_list[i][*iter].length();
			v_min[i]=Min(v_min[i],value);
			v_max[i]=Max(v_max[i],value);
			temp_value = (float)value;
			fwrite(&temp_value, sizeof(float), 1, datafile);
		      }
		      // matrix3 data
		      for(int i = 0; i < (int)md.pv_matrix3_list.size(); i++) {
			double value = md.pv_matrix3_list[i][*iter].Norm();
			m_min[i]=Min(m_min[i],value);
			m_max[i]=Max(m_max[i],value);
			temp_value = (float)value;
			fwrite(&temp_value, sizeof(float), 1, datafile);
		      }
		      if (do_patch) {
			temp_value = (float)patch->getID();
			fwrite(&temp_value, sizeof(float), 1, datafile);
		      }
		      if (do_material) {
			temp_value = (float)m;
			fwrite(&temp_value, sizeof(float), 1, datafile);
		      }
		    }
		  }
		}
	      }
	      
	      //--------------------------------------------------
	      // write the header file

	      if (do_verbose)
		cerr << "---Writing header file\n";
	      if (data_found) {
		fprintf(headerfile,"%d\n",total_particles);
		fprintf(headerfile,"%.17g\n",(max.x()-min.x())/total_particles);
		fprintf(headerfile,"%.17g %.17g\n",min.x(),max.x());
		fprintf(headerfile,"%.17g %.17g\n",min.y(),max.y());
		fprintf(headerfile,"%.17g %.17g\n",min.z(),max.z());
		if (do_PTvar_all) {
		  for(int i = 0; i < (int)d_min.size(); i++) {
		    fprintf(headerfile,"%.17g %.17g\n",d_min[i],d_max[i]);
		  }
		  for(int i = 0; i < (int)f_min.size(); i++) {
		    fprintf(headerfile,"%.17g %.17g\n",f_min[i],f_max[i]);
		  }
		  for(int i = 0; i < (int)v_min.size(); i++) {
		    fprintf(headerfile,"%.17g %.17g\n",v_min[i],v_max[i]);
		  }
		  for(int i = 0; i < (int)m_min.size(); i++) {
		    fprintf(headerfile,"%.17g %.17g\n",m_min[i],m_max[i]);
		  }
		  if (do_patch) {
		    fprintf(headerfile,"%.17g %.17g\n",(float)patch->getID(),(float)patch->getID());
		  }
		  if (do_material) {
		    fprintf(headerfile,"%.17g %.17g\n",0.0,(float)material_data_list.size());
		  }
		}
	      }
	      fclose(datafile);
	      fclose(headerfile);
	    }
	    fprintf(filelist,"</PATCH>\n");
	  } // end patch
	} // end level
	fprintf(filelist,"</TIMESTEP>\n");
      } // end timestep
      fclose(filelist);
    } // end do_rtdata
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}

////////////////////////////////////////////////////////////////////////////
//
// Print a particle variable
//
////////////////////////////////////////////////////////////////////////////
void printParticleVariable(DataArchive* da, 
			   string particleVariable,
			   unsigned long time_step_lower,
			   unsigned long time_step_upper){

  // Check if the particle variable is available
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == particleVariable) variableFound = true;
  }
  if (!variableFound) {
    cerr << "Variable " << particleVariable << " not found\n"; 
    exit(1);
  }

  // Now that the variable has been found, get the data for all 
  // available time steps // from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  //cout << "There are " << index.size() << " timesteps:\n";
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(time);

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patchIterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
	const Patch* patch = *iter;
        ++patchIndex; 

	// Loop thru all the variables 
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];
	  const Uintah::TypeDescription* subtype = td->getSubType();

	  // Check if the variable is a ParticleVariable
	  if(td->getType() == Uintah::TypeDescription::ParticleVariable) { 

	    // loop thru all the materials
	    ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	    ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
	    for(; matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;

	      // Find the name of the variable
	      if (var == particleVariable) {
		//cout << "Material: " << matl << endl;
		switch(subtype->getType()){
		case Uintah::TypeDescription::double_type:
		  {
		    ParticleVariable<double> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl; 
			cout << " " << pid[*iter];
                        cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::float_type:
		  {
		    ParticleVariable<float> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
                        cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::int_type:
		  {
		    ParticleVariable<int> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl;
			cout << " " << pid[*iter];
                        cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Point:
		  {
		    ParticleVariable<Point> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
                        cout << " " << value[*iter](0) 
                             << " " << value[*iter](1)
                             << " " << value[*iter](2) << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Vector:
		  {
		    ParticleVariable<Vector> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
			cout << " " << value[*iter][0] 
                             << " " << value[*iter][1]
                             << " " << value[*iter][2] << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Matrix3:
		  {
		    ParticleVariable<Matrix3> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << pid[*iter];
                        for (int ii = 0; ii < 3; ++ii) {
                          for (int jj = 0; jj < 3; ++jj) {
			    cout << " " << value[*iter](ii,jj) ;
                          }
                        }
			cout << endl;
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::long64_type:
		  {
		    ParticleVariable<long64> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		default:
		  cerr << "Particle Variable of unknown type: " 
		       << subtype->getType() << endl;
		  break;
		}
	      } // end of var compare if
	    } // end of material loop
	  } // end of ParticleVariable if
	} // end of variable loop
      } // end of patch loop
    } // end of level loop
  } // end of time step loop
}

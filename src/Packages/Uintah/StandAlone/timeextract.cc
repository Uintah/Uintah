/*
 *  puda.cc: Print out a uintah data archive
 *
 *  Written by:
 *   James L. Bigler
 *   Bryan J. Worthen
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 U of U
 */

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>

#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

Mutex cerrLock( "cerrLock" );

typedef struct{
  vector<ShareAssignParticleVariable<double> > pv_double_list;
  vector<ShareAssignParticleVariable<Point> > pv_point_list;
  vector<ShareAssignParticleVariable<Vector> > pv_vector_list;
  vector<ShareAssignParticleVariable<Matrix3> > pv_matrix3_list;
  ShareAssignParticleVariable<Point> p_x;
} MaterialData;

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
    cerr << "Usage: " << progname << " [options] -m <material number> "
	 << "-uda <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h[elp]\n";
    cerr << "  -v <variable name>\n";
    cerr << "  -asci\n";
    cerr << "  -verbose (prints status of output)\n";
    cerr << "  -timesteplow [int] (only outputs timestep from int)\n";
    cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
    cerr << "  -index <x> <y> <z> (cell coordinates)\n";
    cerr << "  -o <outputfilename>\n";
    exit(1);
}

// arguments are the dataarchive, the successive arguments are the same as 
// the arguments to archive->query for data values.  Then comes a type 
// dexcription of the variable being queried, and last is an output stream.

void printData(DataArchive* archive, string& variable_name, int material, IntVector& var_id,
               unsigned long time_step_lower, unsigned long time_step_upper,
	       const Uintah::TypeDescription* subtype, ostream& out) 

{
  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
      
  //------------------------------
  // figure out the lower and upper bounds on the timesteps
  if (time_step_lower >= times.size()) {
    cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
    exit(1);
  }

  // set default max time value
  if (time_step_upper == -1)
    time_step_upper = times.size() - 1;

  if (time_step_upper >= times.size() || time_step_upper < time_step_lower) {
    cerr << "timestephigh must be greater than " << time_step_lower 
	 << " and less than " << times.size()-1 << endl;
    exit(1);
  }
  
  double time = times[time_step_lower];
  cout << "time = " << time << endl;
  GridP grid = archive->queryGrid(time);
  
  // set defaults for output stream
  out.setf(ios::scientific,ios::floatfield);
  out.precision(8);
  
  // for each type available, we need to query the values for the time range, 
  // variable name, and material
  switch (subtype->getType()) {
  case Uintah::TypeDescription::double_type:
    {
      vector<double> values;
      try {
	archive->query(values, variable_name, material, var_id, times[time_step_lower], times[time_step_upper]);
      } catch (const VariableNotFoundInGrid& exception) {
	cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	exit(1);
      }
      // Print out data
      for(unsigned int i = 0; i < values.size(); i++) {
	out << values[i] << endl;
      }
    }
  break;
  case Uintah::TypeDescription::int_type:
    {
      vector<int> values;
      try {
	archive->query(values, variable_name, material, var_id, times[time_step_lower], times[time_step_upper]);
      } catch (const VariableNotFoundInGrid& exception) {
	cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	exit(1);
      }
      // Print out data
      for(unsigned int i = 0; i < values.size(); i++) {
	out << values[i] << endl;
      }
    }
  break;
  default:
    {
      cerr << "Unknown subtype\n";
      exit(1);
    }
  }
} 

int main(int argc, char** argv)
{
  /*
   * Default values
   */
  bool do_asci=false;
  bool do_verbose = false;
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = -1;  // default to be last timestep, but can
                                       // be set to 0
  string input_uda_name;
  string output_file_name("-");
  IntVector var_id(0,0,0);
  string variable_name;
  // Now the material index is kind of a hard thing.  There is no way
  // to reliably determine a default material.  Materials are defined
  // on the patch for each varialbe, so this subset of materials could
  // change over patches.  Because of this we will force the user to
  // specify the material.
  int material = -1;
  
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-asci"){
      do_asci=true;
    } else if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if (s == "-m" || s == "-material") {
      material = atoi(argv[++i]);
    } else if (s == "-verbose") {
      do_verbose = true;
    } else if (s == "-timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-index" || s == "-i") {
      unsigned long x = strtoul(argv[++i],(char**)NULL,10);
      unsigned long y = strtoul(argv[++i],(char**)NULL,10);
      unsigned long z = strtoul(argv[++i],(char**)NULL,10);
      var_id = IntVector(x,y,z);
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
    } else {
      usage(s, argv[0]);
    }
  }
  
  if(input_uda_name == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  if (material == -1) {
    cerr << "Must specify material number.\n";
    usage("", argv[0]);
  }
  
  try {
    DataArchive* archive = scinew DataArchive(input_uda_name);
    
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    cout << "There are " << vars.size() << " variables:\n";
    bool var_found = false;
    unsigned int var_index = 0;
    for (;var_index < vars.size(); var_index++) {
      var_found = true;
      break;
    }
    
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
      //      var = vars[0];
    }

    cout << vars[var_index] << ": " << types[var_index]->getName() << endl;

    // get type and subtype of data
    const Uintah::TypeDescription* td = types[var_index];
    const Uintah::TypeDescription* subtype = td->getSubType();

    // Open output file, call printData with it's ofstream
    // if no output file, call with cout
    if (output_file_name != "-") {
      ofstream output;
      output.open(output_file_name.c_str());
      if (!output) {
	// Error!!
	cerr << "Could not open "<<output_file_name<<" for writing.\n";
	exit(1);
      }
      printData(archive, variable_name, material, var_id, time_step_lower,
                time_step_upper, subtype, output);
    } else {
      printData(archive, variable_name, material, var_id, time_step_lower,
              time_step_upper, subtype, cout);
    }
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    exit(1);
  } catch(...){
    cerr << "Caught unknown exception\n";
    exit(1);
  }
}

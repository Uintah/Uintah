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

bool verbose = false;
bool quite = false;

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
	cerr << "Error parsing argument: " << badarg << endl;
    cerr << "Usage: " << progname << " [options] "
	 << "-uda <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h,--help\n";
    cerr << "  -v,--variable <variable name>\n";
    cerr << "  -m,--material <material number> [defaults to 0]\n";
    //    cerr << "  -binary (prints out the data in binary)\n";
    cerr << "  -tlow,--timesteplow [int] (only outputs timestep from int) [defaults to 0]\n";
    cerr << "  -thigh,--timestephigh [int] (only outputs timesteps up to int) [defaults to last timestep]\n";
    cerr << "  -i,--index <x> <y> <z> (cell coordinates) [defaults to 0,0,0]\n";
    cerr << "  -o,--out <outputfilename> [defaults to stdout]\n";
    cerr << "  -vv,--verbose (prints status of output)\n";
    cerr << "  -q,--quite (only print data values)\n";
    exit(1);
}

// arguments are the dataarchive, the successive arguments are the same as 
// the arguments to archive->query for data values.  Then comes a type 
// dexcription of the variable being queried, and last is an output stream.

template<class T>
void printData(DataArchive* archive, string& variable_name,
	       int material, IntVector& var_id,
               unsigned long time_step_lower, unsigned long time_step_upper,
	       ostream& out) 

{
  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  if (!quite) cout << "There are " << index.size() << " timesteps:\n";
      
  //------------------------------
  // figure out the lower and upper bounds on the timesteps
  if (time_step_lower >= times.size()) {
    cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
    exit(1);
  }

  // set default max time value
  if (time_step_upper == (unsigned long)-1) {
    if (verbose)
      cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
    time_step_upper = times.size() - 1;
  }

  if (time_step_upper >= times.size() || time_step_upper < time_step_lower) {
    cerr << "timestephigh("<<time_step_lower<<") must be greater than " << time_step_lower 
	 << " and less than " << times.size()-1 << endl;
    exit(1);
  }
  
  if (!quite) cout << "outputting for times["<<time_step_lower<<"] = " << times[time_step_lower]<<" to times["<<time_step_upper<<"] = "<<times[time_step_upper] << endl;
  
  // set defaults for output stream
  out.setf(ios::scientific,ios::floatfield);
  out.precision(8);
  
  // for each type available, we need to query the values for the time range, 
  // variable name, and material
  vector<T> values;
  try {
    archive->query(values, variable_name, material, var_id, times[time_step_lower], times[time_step_upper]);
  } catch (const VariableNotFoundInGrid& exception) {
    cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
    exit(1);
  }
  // Print out data
  for(unsigned int i = 0; i < values.size(); i++) {
    out << times[i] << ", " << values[i] << endl;
  }
} 

int main(int argc, char** argv)
{
  /*
   * Default values
   */
  bool do_binary=false;

  unsigned long time_step_lower = 0;
  // default to be last timestep, but can be set to 0
  unsigned long time_step_upper = (unsigned long)-1;

  string input_uda_name;
  string output_file_name("-");
  IntVector var_id(0,0,0);
  string variable_name;
  // Now the material index is kind of a hard thing.  There is no way
  // to reliably determine a default material.  Materials are defined
  // on the patch for each varialbe, so this subset of materials could
  // change over patches.  We can guess, that there will be a material
  // 0.  This shouldn't cause the program to crash.  It will spit out
  // an exception and exit gracefully.
  int material = 0;
  
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if (s == "-m" || s == "--material") {
      material = atoi(argv[++i]);
    } else if (s == "-vv" || s == "--verbose") {
      verbose = true;
    } else if (s == "-q" || s == "--quite") {
      quite = true;
    } else if (s == "-tlow" || s == "--timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-thigh" || s == "--timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-i" || s == "--index") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_id = IntVector(x,y,z);
    } else if( (s == "-h") || (s == "--help") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
    } else if(s == "-binary") {
      do_binary=true;
    } else {
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

    if (!quite) cout << vars[var_index] << ": " << types[var_index]->getName() << " being extracted for material "<<material<<" at index "<<var_id<<endl;

    // get type and subtype of data
    const Uintah::TypeDescription* td = types[var_index];
    const Uintah::TypeDescription* subtype = td->getSubType();

    // Open output file, call printData with it's ofstream
    // if no output file, call with cout
    ostream *output_stream = &cout;
    if (output_file_name != "-") {
      if (verbose) cout << "Opening \""<<output_file_name<<"\" for writing.\n";
      ofstream *output = new ofstream();
      output->open(output_file_name.c_str());
      if (!(*output)) {
	// Error!!
	cerr << "Could not open "<<output_file_name<<" for writing.\n";
	exit(1);
      }
      output_stream = output;
    } else {
      //output_stream = cout;
    }
  switch (subtype->getType()) {
  case Uintah::TypeDescription::double_type:
    printData<double>(archive, variable_name, material, var_id,
		      time_step_lower, time_step_upper, *output_stream);
    break;
  case Uintah::TypeDescription::int_type:
    printData<int>(archive, variable_name, material, var_id,
		   time_step_lower, time_step_upper, *output_stream);
    break;
  case Uintah::TypeDescription::Vector:
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

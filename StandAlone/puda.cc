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

#include <Packages/Uintah/CCA/Ports/DataArchive.h>
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
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>


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
	cerr << "Error parsing argument: " << badarg << '\n';
    cerr << "Usage: " << progname << " [options] <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h[elp]\n";
    cerr << "  -timesteps\n";
    cerr << "  -gridstats\n";
    cerr << "  -listvariables\n";
    cerr << "  -varsummary\n";
    cerr << "  -asci\n";
    cerr << "  -cell_stresses\n";
    cerr << "  -rtdata [output directory]\n";
    cerr << "  -PTvar\n";
    cerr << "  -ptonly (prints out only the point location\n";
    cerr << "  -patch (outputs patch id with data)\n";
    cerr << "  -material (outputs material number with data)\n";
    cerr << "  -NCvar [double or point or vector]\n";
    cerr << "  -CCvar [double or point or vector]\n";
    cerr << "  -verbose (prints status of output)\n";
    cerr << "  -timesteplow [int] (only outputs timestep from int)\n";
    cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
    cerr << "*NOTE* to use -PTvar or -NVvar -rtdata must be used\n";
    cerr << "*NOTE* ptonly, patch, material, timesteplow, timestephigh \
are used in conjuntion with -PTvar.\n\n";
    
    cerr << "USAGE IS NOT FINISHED\n\n";
    exit(1);
}

int main(int argc, char** argv)
{
  /*
   * Default values
   */
  bool do_timesteps=false;
  bool do_gridstats=false;
  bool do_listvars=false;
  bool do_varsummary=false;
  bool do_asci=false;
  bool do_cell_stresses=false;
  bool do_rtdata = false;
  bool do_NCvar_double = false;
  bool do_NCvar_point = false;
  bool do_NCvar_vector = false;
  bool do_NCvar_matrix3 = false;
  bool do_CCvar_double = false;
  bool do_CCvar_point = false;
  bool do_CCvar_vector = false;
  bool do_CCvar_matrix3 = false;
  bool do_PTvar = false;
  bool do_PTvar_all = true;
  bool do_patch = false;
  bool do_material = false;
  bool do_verbose = false;
  unsigned long time_step_lower;
  unsigned long time_step_upper;
  bool tslow_set = false;
  bool tsup_set = false;
  string filebase;
  string raydatadir;
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-timesteps"){
      do_timesteps=true;
    } else if(s == "-gridstats"){
      do_gridstats=true;
    } else if(s == "-listvariables"){
      do_listvars=true;
    } else if(s == "-varsummary"){
      do_varsummary=true;
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
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
    } else {
      if(filebase!="")
	usage(s, argv[0]);
      else
	filebase = argv[i];
    }
  }
  
  if(filebase == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    XMLPlatformUtils::Initialize();
  } catch(const XMLException& toCatch) {
    cerr << "Caught XML exception: " << toString(toCatch.getMessage()) 
	 << '\n';
    exit( 1 );
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
	cout << index[i] << ": " << times[i] << '\n';
    }
    if(do_gridstats){
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";
      for(int i=0;i<(int)index.size();i++){
	cout << index[i] << ": " << times[i] << '\n';
	GridP grid = da->queryGrid(times[i]);
	grid->performConsistencyCheck();
	grid->printStatistics();
      }
    }
    if(do_listvars){
      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;
      da->queryVariables(vars, types);
      cout << "There are " << vars.size() << " variables:\n";
      for(int i=0;i<(int)vars.size();i++){
	cout << vars[i] << ": " << types[i]->getName() << '\n';
      }
    }
    if(do_varsummary){
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
      
      for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
	double time = times[t];
	cout << "time = " << time << "\n";
	GridP grid = da->queryGrid(time);
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];
	  const Uintah::TypeDescription* subtype = td->getSubType();
	  cout << "\tVariable: " << var << ", type " << td->getName() << "\n";
	  for(int l=0;l<grid->numLevels();l++){
	    LevelP level = grid->getLevel(l);
	    for(Level::const_patchIterator iter = level->patchesBegin();
		iter != level->patchesEnd(); iter++){
	      const Patch* patch = *iter;
	      cout << "\t\tPatch: " << patch->getID() << "\n";
	      ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	      // loop over materials
	      for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		  matlIter != matls.end(); matlIter++){
		int matl = *matlIter;
		cout << "\t\t\tMaterial: " << matl << "\n";
		switch(td->getType()){
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
			cout << "\t\t\t\tmin value: " << min << '\n';
			cout << "\t\t\t\tmax value: " << max << '\n';
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
			 cout << "\t\t\t\tmin value: " << min << '\n';
			cout << "\t\t\t\tmax value: " << max << '\n';
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
			cout << "\t\t\t\tmin magnitude: " << sqrt(min) << '\n';
			cout << "\t\t\t\tmax magnitude: " << sqrt(max) << '\n';
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
			cout << "\t\t\t\tmin Norm: " << min << '\n';
			cout << "\t\t\t\tmax Norm: " << max << '\n';
		      }
		    }
		  break;
		  default:
		    cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
		    break;
		  }
		  break;
		case Uintah::TypeDescription::NCVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      NCVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << '\n';
			cout << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      NCVariable<Point> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			Point min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << '\n';
			cout << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      NCVariable<Vector> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter].length2();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].length2());
			  max=Max(max, value[*iter].length2());
			}
			cout << "\t\t\t\tmin magnitude: " << sqrt(min) << '\n';
			cout << "\t\t\t\tmax magnitude: " << sqrt(max) << '\n';
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Matrix3:
		    {
		      NCVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter].Norm();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].Norm());
			  max=Max(max, value[*iter].Norm());
			}
			cout << "\t\t\t\tmin Norm: " << min << '\n';
			cout << "\t\t\t\tmax Norm: " << max << '\n';
		      }
		    }
		  break;
		  default:
		    cerr << "NC Variable of unknown type: " << subtype->getType() << '\n';
		    break;
		  }
		  break;
		case Uintah::TypeDescription::CCVariable:
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      CCVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << '\n';
			cout << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      CCVariable<Point> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			Point min, max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cout << "\t\t\t\tmin value: " << min << '\n';
			cout << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      CCVariable<Vector> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter].length2();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].length2());
			  max=Max(max, value[*iter].length2());
			}
			cout << "\t\t\t\tmin magnitude: " << sqrt(min) << '\n';
			cout << "\t\t\t\tmax magnitude: " << sqrt(max) << '\n';
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Matrix3:
		    {
		      CCVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			CellIterator iter = patch->getCellIterator();
			min=max=value[*iter].Norm();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].Norm());
			  max=Max(max, value[*iter].Norm());
			}
			cout << "\t\t\t\tmin Norm: " << min << '\n';
			cout << "\t\t\t\tmax Norm: " << max << '\n';
		      }
		    }
		  break;
		  default:
		    cerr << "CC Variable of unknown type: " << subtype->getType() << '\n';
		    break;
		  }
		  break;
		default:
		  cerr << "Variable of unknown type: " << td->getType() << '\n';
		  break;
		}
	      }
	    }
	  }
	}
      }
    }
    
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
      
      // Loop over time
      for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
	double time = times[t];
      	int partnum = 1;
	int num_of_particles = 0;
	cout << "timestep " << ts << " inprogress... ";
	
  	if (( ts % freq) == 0) {
   		
		// dumps header and variable info to file
		//int variable_count =0;
		char fnum[5];
   		string filename;
   		int stepnum=ts/freq;
	        sprintf(fnum,"%04d",stepnum);
   		string partroot("partout");
                filename = partroot+fnum;
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
		     cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
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
			    partfile << (value[*iter])(1,3) << " " << endl;
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
			  iter = pset->begin();
			  for(;iter !=pset->end(); iter++){
			    partfile << (value[*iter])(2,3) << " " << endl;
			  }
			  partfile << endl;
			  iter = pset->begin();
			  for(;iter !=pset->end(); iter++){
			    partfile << (value[*iter])(3,1) << " " << endl;
			  }
			  partfile << endl;
			  iter = pset->begin();
			  for(;iter !=pset->end(); iter++){
			    partfile << (value[*iter])(3,2) << " " << endl;
			  }
			  partfile << endl;
			  iter = pset->begin();
			  for(;iter !=pset->end(); iter++){
			    partfile << (value[*iter])(3,3) << " " << endl;
			  }
			  partfile << endl;
			}
		      }
		      break;
		    default:
		      cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
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
      
      // obtain the desired timesteps
      unsigned int t = 0, start_time, stop_time;

      cout << "Time Step       Value\n";
      
      for(t = time_step_lower; t <= time_step_upper; t++){
	double time = times[t];
	cout << "    " << t + 1 << "        "  << time << "\n";
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
	cout << "time = " << time << "\n";
	GridP grid = da->queryGrid(time);
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  
	  // only dumps out data if it is variable g.stressFS
	  if (var == "g.stressFS"){
	    const Uintah::TypeDescription* td = types[v];
	    const Uintah::TypeDescription* subtype = td->getSubType();
	    cout << "\tVariable: " << var << ", type " << td->getName() << "\n";
	    for(int l=0;l<grid->numLevels();l++){
	      LevelP level = grid->getLevel(l);
	      for(Level::const_patchIterator iter = level->patchesBegin();
		  iter != level->patchesEnd(); iter++){
		const Patch* patch = *iter;
		cout << "\t\tPatch: " << patch->getID() << "\n";
                ConsecutiveRangeSet matls =
		   da->queryMaterials(var, patch, time);
	        // loop over materials
	        for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		    matlIter != matls.end(); matlIter++){
		  int matl = *matlIter;
		  
		  // dumps header and variable info to file
		  char fnum[5], matnum[5];
		  string filename;
		  int timestepnum=t+1;
		  sprintf(fnum,"%04d",timestepnum);
		  sprintf(matnum,"%04d",matl);
		  string partroot("stress.t");
		  string partext(".m");
		  filename = partroot+fnum+partext+matnum;
		  ofstream partfile(filename.c_str());
		  partfile << "# x, y, z, st11, st12, st13, st21, st22, st23, st31, st32, st33" << endl;
		  
		  cout << "\t\t\tMaterial: " << matl << "\n";
		  switch(td->getType()){
		  case Uintah::TypeDescription::NCVariable:
		    switch(subtype->getType()){
		    case Uintah::TypeDescription::Matrix3:{
		      NCVariable<Matrix3> value;
		      da->query(value, var, matl, patch, time);
		      cout << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex()
			   << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			NodeIterator iter = patch->getNodeIterator();
			for(;!iter.done(); iter++){
			  partfile << (*iter).x() << " " << (*iter).y() << " " << (*iter).z()
				   << " " << (value[*iter])(1,1) << " " << (value[*iter])(1,2) << " " 
				   << (value[*iter])(1,3) << " " << (value[*iter])(2,1) << " "
				   << (value[*iter])(2,2) << " " << (value[*iter])(2,3) << " "
				   << (value[*iter])(3,1) << " " << (value[*iter])(3,2) << " "
				   << (value[*iter])(3,3) << endl;
			}
		      }
		    }
		    break;
		    default:
		      cerr << "No Matrix3 Subclass avaliable." << subtype->getType() << '\n';
		      break;
		    }
		    break;
		  default:
		    cerr << "No NC Variables avaliable." << td->getType() << '\n';
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
	  cerr << "Caught exception: " << e.message() << '\n';
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
      
      // for all timesteps
      for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
	double time = times[t];
	ostringstream tempstr_time;
	tempstr_time << setprecision(17) << time;
	time_file = replaceChar(string(tempstr_time.str()),'.','_');
	GridP grid = da->queryGrid(time);
	fprintf(filelist,"<TIMESTEP>\n");
	if(do_verbose)
	  cout << "time = " << time << "\n";
	// Create a directory if it's not already there.
	// The exception occurs when the directory is already there
	// and the Dir.create fails.  This exception is ignored. 
	Dir rayDir;
	try {
	  rayDir.create(raydatadir + string("/TS_") + time_file);
	}
	catch (Exception& e) {
	  cerr << "Caught directory making exception: " << e.message() << '\n';
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
		    case Uintah::TypeDescription::Point:
		      {
			ParticleVariable<Point> value;
			da->query(value, var, matl, patch, time);
			
			if (var == "p.x") {
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
		      cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
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
			double min, max;
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
		    cerr << "NC variable of unknown type: " << subtype->getType() << '\n';
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
			double min, max;
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
		    cerr << "CC variable of unknown type: " << subtype->getType() << '\n';
		    break;
		  }
		  break;
		default:
		  cerr << "Variable of unknown type: " << td->getType() << '\n';
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
	      vector<double> d_min,d_max,v_min,v_max,m_min,m_max;
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
		  cerr << "No particle location variable found\n";
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
		  cerr << "No particle location variable found\n";
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
    cerr << "Caught exception: " << e.message() << '\n';
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}

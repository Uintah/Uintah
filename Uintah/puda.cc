
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

#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/NodeIterator.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/OS/Dir.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>


using SCICore::Exceptions::Exception;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::OS;
using namespace std;
using namespace Uintah;
using namespace PSECore::XMLUtil;

// takes a string and replaces all occurances of old with newch
string replaceChar(string s, char old, char newch) {
  string result;
  for (int i = 0; i<s.size(); i++)
    if (s[i] == old)
      result += newch;
    else
      result += s[i];
  return result;
}

// use this function to open a pair of files for outputing
// data to the reat-time raytracer.
//
// in: pointers to the pointer to the file
//     the file name
// out: inialized files for writing
//      boolean reporting the success of the file creation
bool setupOutFiles(FILE** data, FILE** header, string name) {
  FILE* datafile;
  FILE* headerfile;
  string headername = name + string(".hdr");

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

  string raydatafile = raydatadir + string("/");
  raydatafile+= string("VAR_") + variable_file + string(".");
  raydatafile+= string("TS_") + time_file + string(".");
  raydatafile+= string("PI_") + patchID_file + string(".");
  raydatafile+= string("MT_") + materialType_file;
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
    cerr << "  -rtdata [output directory]\n";
    cerr << "  -PTvar [double or point or vector]\n";
    cerr << "  -NCvar [double or point or vector]\n";
    cerr << "*NOTE* to use -PTvar or -NVvar -rtdata must be used\n\n";

    cerr << "*NOTE* currently only -NCvar double is implemented\n";
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
  bool do_rtdata = false;
  bool do_NCvar_double = false;
  bool do_NCvar_point = false;
  bool do_NCvar_vector = false;
  bool do_PTvar_double = false;
  bool do_PTvar_point = false;
  bool do_PTvar_vector = false;
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
	else
	  usage("-NCvar", argv[0]);
      }
      else
	usage("-NCvar", argv[0]);
    } else if(s == "-PTvar") {
      if (++i < argc) {
	s = argv[i];
	if (s == "double")
	  do_PTvar_double = true;
	else if (s == "point")
	  do_PTvar_point = true;
	else if (s == "vector")
	  do_PTvar_vector = true;
	else
	  usage("-PTvar", argv[0]);
      }
      else
	usage("-PTvar", argv[0]);
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
  
  try {
    XMLPlatformUtils::Initialize();
  } catch(const XMLException& toCatch) {
    cerr << "Caught XML exception: " << toString(toCatch.getMessage()) 
	 << '\n';
    exit( 1 );
  }
  
  try {
    DataArchive* da = new DataArchive(filebase);
    
    if(do_timesteps){
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";
      for(int i=0;i<index.size();i++)
	cout << index[i] << ": " << times[i] << '\n';
    }
    if(do_gridstats){
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";
      for(int i=0;i<index.size();i++){
	cout << index[i] << ": " << times[i] << '\n';
	GridP grid = da->queryGrid(times[i]);
	grid->performConsistencyCheck();
	grid->printStatistics();
      }
    }
    if(do_listvars){
      vector<string> vars;
      vector<const TypeDescription*> types;
      da->queryVariables(vars, types);
      cout << "There are " << vars.size() << " variables:\n";
      for(int i=0;i<vars.size();i++){
	cout << vars[i] << ": " << types[i]->getName() << '\n';
      }
    }
    if(do_varsummary){
      vector<string> vars;
      vector<const TypeDescription*> types;
      da->queryVariables(vars, types);
      ASSERTEQ(vars.size(), types.size());
      cout << "There are " << vars.size() << " variables:\n";
      
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      cout << "There are " << index.size() << " timesteps:\n";
      
      for(int t=0;t<times.size();t++){
	double time = times[t];
	cerr << "time = " << time << "\n";
	GridP grid = da->queryGrid(time);
	for(int v=0;v<vars.size();v++){
	  std::string var = vars[v];
	  const TypeDescription* td = types[v];
	  const TypeDescription* subtype = td->getSubType();
	  cerr << "\tVariable: " << var << ", type " << td->getName() << "\n";
	  for(int l=0;l<grid->numLevels();l++){
	    LevelP level = grid->getLevel(l);
	    for(Level::const_patchIterator iter = level->patchesBegin();
		iter != level->patchesEnd(); iter++){
	      const Patch* patch = *iter;
	      cerr << "\t\tPatch: " << patch->getID() << "\n";
	      int numMatls = da->queryNumMaterials(var, patch, time);
	      for(int matl=0;matl<numMatls;matl++){
		cerr << "\t\t\tMaterial: " << matl << "\n";
		switch(td->getType()){
		case TypeDescription::ParticleVariable:
		  switch(subtype->getType()){
		  case TypeDescription::double_type:
		    {
		      ParticleVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cerr << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			double min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++];
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cerr << "\t\t\t\tmin value: " << min << '\n';
			cerr << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case TypeDescription::Point:
		    {
		      ParticleVariable<Point> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cerr << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			Point min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++];
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cerr << "\t\t\t\tmin value: " << min << '\n';
			cerr << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case TypeDescription::Vector:
		    {
		      ParticleVariable<Vector> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      cerr << "\t\t\t\t" << td->getName() << " over " << pset->numParticles() << " particles\n";
		      if(pset->numParticles() > 0){
			double min, max;
			ParticleSubset::iterator iter = pset->begin();
			min=max=value[*iter++].length2();
			for(;iter != pset->end(); iter++){
			  min=Min(min, value[*iter].length2());
			  max=Max(max, value[*iter].length2());
			}
			cerr << "\t\t\t\tmin magnitude: " << sqrt(min) << '\n';
			cerr << "\t\t\t\tmax magnitude: " << sqrt(max) << '\n';
		      }
		    }
		  break;
		  default:
		    cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
		    break;
		  }
		  break;
		case TypeDescription::NCVariable:
		  switch(subtype->getType()){
		  case TypeDescription::double_type:
		    {
		      NCVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      cerr << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cerr << "\t\t\t\tmin value: " << min << '\n';
			cerr << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case TypeDescription::Point:
		    {
		      NCVariable<Point> value;
		      da->query(value, var, matl, patch, time);
		      cerr << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			Point min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter];
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter]);
			  max=Max(max, value[*iter]);
			}
			cerr << "\t\t\t\tmin value: " << min << '\n';
			cerr << "\t\t\t\tmax value: " << max << '\n';
		      }
		    }
		  break;
		  case TypeDescription::Vector:
		    {
		      NCVariable<Vector> value;
		      da->query(value, var, matl, patch, time);
		      cerr << "\t\t\t\t" << td->getName() << " over " << value.getLowIndex() << " to " << value.getHighIndex() << "\n";
		      IntVector dx(value.getHighIndex()-value.getLowIndex());
		      if(dx.x() && dx.y() && dx.z()){
			double min, max;
			NodeIterator iter = patch->getNodeIterator();
			min=max=value[*iter].length2();
			for(;!iter.done(); iter++){
			  min=Min(min, value[*iter].length2());
			  max=Max(max, value[*iter].length2());
			}
			cerr << "\t\t\t\tmin magnitude: " << sqrt(min) << '\n';
			cerr << "\t\t\t\tmax magnitude: " << sqrt(max) << '\n';
		      }
		    }
		  break;
		  default:
		    cerr << "NC Variable of unknown type: " << subtype->getType() << '\n';
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
    if (do_rtdata) {
      vector<string> vars;
      vector<const TypeDescription*> types;
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
      
      for(int t=0;t<times.size();t++){
	double time = times[t];
	ostringstream tempstr_time;
	tempstr_time << time;
	time_file = replaceChar(string(tempstr_time.str()),'.','_');
	GridP grid = da->queryGrid(time);
	
	for(int v=0;v<vars.size();v++){
	  std::string var = vars[v];
	  variable_file = replaceChar(var,'.','_');
	  const TypeDescription* td = types[v];
	  const TypeDescription* subtype = td->getSubType();
	  
	  for(int l=0;l<grid->numLevels();l++){
	    LevelP level = grid->getLevel(l);
	    
	    for(Level::const_patchIterator iter = level->patchesBegin();
		iter != level->patchesEnd(); iter++){
	      const Patch* patch = *iter;
	      ostringstream tempstr_patch;
	      tempstr_patch << patch->getID();
	      patchID_file = tempstr_patch.str();
	      int numMatls = da->queryNumMaterials(var, patch, time);
	      
	      for(int matl=0;matl<numMatls;matl++){
		ostringstream tempstr_matl;
		tempstr_matl << matl;
		materialType_file = tempstr_matl.str();

	        switch(td->getType()){
	        case TypeDescription::ParticleVariable:
		  switch(subtype->getType()){
		  case TypeDescription::double_type:
		    {
		      if (do_PTvar_double) {
			// not implemented at this time
		      }
		    }
		  break;
		  case TypeDescription::Point:
		    {
		      if (do_PTvar_point) {
			// not implemented at this time
		      }
		    }
		  break;
		  case TypeDescription::Vector:
		    {
		      if (do_PTvar_vector) {
			// not implemented at this time
		      }
		    }
		  break;
		  default:
		    cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
		    break;
		  }
		  break;
		case TypeDescription::NCVariable:
		  switch(subtype->getType()){
		  case TypeDescription::double_type:
		    {
		      if (do_NCvar_double) {
			// setup output files
			string raydatafile = makeFileName(raydatadir,variable_file,time_file,patchID_file,materialType_file);			
			FILE* datafile;
			FILE* headerfile;
			if (!setupOutFiles(&datafile,&headerfile,raydatafile))
			  abort();

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
		  case TypeDescription::Point:
		    {
		      if (do_NCvar_point) {
			// not implemented at this time
		      }
		    }
		  break;
		  case TypeDescription::Vector:
		    {
		      if (do_NCvar_vector) {
			// not implemented at this time
		      }
		    }
		  break;
		  default:
		    cerr << "NC variable of unknown type: " << subtype->getType() << '\n';
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
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}

//
// $Log$
// Revision 1.5  2000/06/08 20:58:42  bigler
// Added support to ouput data for the reat-time raytracer.
//
// Revision 1.4  2000/05/30 20:18:39  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.3  2000/05/21 08:19:04  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.2  2000/05/20 19:54:52  dav
// browsing puda, added a couple of things to usage, etc.
//
// Revision 1.1  2000/05/20 08:09:01  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//

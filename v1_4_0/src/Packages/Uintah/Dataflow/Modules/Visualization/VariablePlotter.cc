/*
 *  VariablePlotter.cc:  Displays Patch boundaries
 *
 *  This module is designed to allow the user to select a variable by the
 *  index and display the value over time in a graph or table.
 *
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h> // Includ after Patch.h
#include <Packages/Uintah/Core/Grid/CellIterator.h> // Includ after Patch.h
//#include <Packages/Uintah/Core/Grid/FaceIterator.h> // Includ after Patch.h
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <vector>
#include <sstream>
#include <iostream>
//#include <string>

using namespace SCIRun;
using namespace std;


namespace Uintah {

// should match the values in the tcl code
#define NC_VAR 0
#define CC_VAR 1

class ID {
public:
  ID(): id(IntVector(0,0,0)), level(0) {}
  IntVector id;
  int level;
};
  
class VariablePlotter : public Module {
public:
  VariablePlotter(const string& id);
  virtual ~VariablePlotter();
  virtual void execute();
  void tcl_command(TCLArgs& args, void* userdata);

private:
  GridP getGrid();
  void add_type(string &type_list,const TypeDescription *subtype);
  void setVars(GridP grid);
  void extract_data(string display_mode, string varname,
		    vector<string> mat_list, vector<string> type_list,
		    string index);
  void pick(); // synchronize user set index values with currentNode
  
  // caching variables
  bool is_cached(string name, string& data);
  void cache_value(string where, vector<double>& values, string &data);
  void cache_value(string where, vector<Vector>& values);
  void cache_value(string where, vector<Matrix3>& values);
  string vector_to_string(vector< int > data);
  string vector_to_string(vector< string > data);
  string vector_to_string(vector< double > data);
  string vector_to_string(vector< Vector > data, string type);
  string vector_to_string(vector< Matrix3 > data, string type);
  string currentNode_str();
  
  ArchiveIPort* in; // incoming data archive

  GuiInt var_orientation; // whether node center or cell centered
  GuiInt nl;      // number of levels in the scene
  GuiInt index_x; // x index for the variable
  GuiInt index_y; // y index for the variable
  GuiInt index_z; // z index for the variable
  GuiInt index_l; // index of the level for the variable
  GuiString curr_var;
  
  ID currentNode;
  vector< string > names;
  vector< double > times;
  vector< const TypeDescription *> types;
  double time;
  DataArchive* archive;
  map< string, string > material_data_list;
};

static string widget_name("VariablePlotter Widget");
 
extern "C" Module* make_VariablePlotter(const string& id) {
  return scinew VariablePlotter(id);
}

VariablePlotter::VariablePlotter(const string& id)
: Module("VariablePlotter", id, Filter, "Visualization", "Uintah"),
  var_orientation("var_orientation",id,this),
  nl("nl",id,this),
  index_x("index_x",id,this),
  index_y("index_y",id,this),
  index_z("index_z",id,this),
  index_l("index_l",id,this),
  curr_var("curr_var",id,this)
{

}

VariablePlotter::~VariablePlotter()
{
}

// returns a pointer to the grid
GridP VariablePlotter::getGrid()
{
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"Didn't get a handle\n";
    return 0;
  }

  // access the grid through the handle and dataArchive
  archive = (*(handle.get_rep()))();
  vector< int > indices;
  times.clear();
  archive->queryTimesteps( indices, times );
  TCL::execute(id + " set_time " + vector_to_string(indices).c_str());
  int timestep = (*(handle.get_rep())).timestep();
  time = times[timestep];
  GridP grid = archive->queryGrid(time);

  return grid;
}

void VariablePlotter::add_type(string &type_list,const TypeDescription *subtype)
{
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
    type_list += " scaler";
    break;
  case TypeDescription::Vector:
    type_list += " vector";
    break;
  case TypeDescription::Matrix3:
    type_list += " matrix3";
    break;
  default:
    cerr<<"Error in VariablePlotter::setVars(): Vartype not implemented.  Aborting process.\n";
    abort();
  }
}  

void VariablePlotter::setVars(GridP grid) {
  names.clear();
  types.clear();
  archive->queryVariables(names, types);

  string varNames("");
  string type_list("");
  const Patch* patch = *(grid->getLevel(0)->patchesBegin());

  cerr << "Calling clearMat_list\n";
  TCL::execute(id + " clearMat_list ");
  
  for(int i = 0; i< (int)names.size(); i++) {
    switch (types[i]->getType()) {
    case TypeDescription::NCVariable:
      if (var_orientation.get() == NC_VAR) {
	varNames += " ";
	varNames += names[i];
	cerr << "Calling appendMat_list\n";
	TCL::execute(id + " appendMat_list " + archive->queryMaterials(names[i], patch, time).expandedString().c_str());
	add_type(type_list,types[i]->getSubType());
      }
      break;
    case TypeDescription::CCVariable:
      if (var_orientation.get() == CC_VAR) {
	varNames += " ";
	varNames += names[i];
	cerr << "Calling appendMat_list\n";
	TCL::execute(id + " appendMat_list " + archive->queryMaterials(names[i], patch, time).expandedString().c_str());
	add_type(type_list,types[i]->getSubType());
      }
      break;
    default:
      cerr << "VariablePlotter::setVars: Warning!  Ignoring unknown type.\n";
      break;
    }

    
  }

  cerr << "varNames = " << varNames << endl;
  TCL::execute(id + " setVar_list " + varNames.c_str());
  TCL::execute(id + " setType_list " + type_list.c_str());  
}

void VariablePlotter::execute()
{
  // Create the input port
  in= (ArchiveIPort *) get_iport("Data Archive");

  cerr << "\t\tEntering execute.\n";

  // Get the handle on the grid and the number of levels
  GridP grid = getGrid();
  if(!grid)
    return;
  int numLevels = grid->numLevels();

  // setup the tickle stuff
  nl.set(numLevels);
  setVars(grid);
  string visible;
  TCL::eval(id + " isVisible", visible);
  if ( visible == "1") {
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");

    TCL::execute("update idletasks");
    reset_vars();
  }
  
  cerr << "\t\tFinished execute\n";
}

void VariablePlotter::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  else if(args[1] == "pick") {
    pick();
  }
  else if(args[1] == "extract_data") {
    int i = 2;
    string displaymode(args[i++]);
    string varname(args[i++]);
    string index(args[i++]);
    int num_mat;
    string_to_int(args[i++], num_mat);
    cerr << "Extracting " << num_mat << " materals:";
    vector< string > mat_list;
    vector< string > type_list;
    for (int j = i; j < i+(num_mat*2); j++) {
      string mat(args[j]);
      mat_list.push_back(mat);
      j++;
      string type(args[j]);
      type_list.push_back(type);
    }
    cerr << endl;
    cerr << "Graphing " << varname << " with materials: " << vector_to_string(mat_list) << endl;
    extract_data(displaymode,varname,mat_list,type_list,index);
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

bool VariablePlotter::is_cached(string name, string& data) {
  map< string, string >::iterator iter;
  iter = material_data_list.find(name);
  if (iter == material_data_list.end()) {
    return false;
  }
  else {
    data = iter->second;
    return true;
  }
}

void VariablePlotter::cache_value(string where, vector<double>& values,
				 string &data) {
  data = vector_to_string(values);
  material_data_list[where] = data;
}

void VariablePlotter::cache_value(string where, vector<Vector>& values) {
  string data = vector_to_string(values,"length");
  material_data_list[where+" length"] = data;
  data = vector_to_string(values,"length2");
  material_data_list[where+" length2"] = data;
  data = vector_to_string(values,"x");
  material_data_list[where+" x"] = data;
  data = vector_to_string(values,"y");
  material_data_list[where+" y"] = data;
  data = vector_to_string(values,"z");
  material_data_list[where+" z"] = data;
}

void VariablePlotter::cache_value(string where, vector<Matrix3>& values) {
  string data = vector_to_string(values,"Determinant");
  material_data_list[where+" Determinant"] = data;
  data = vector_to_string(values,"Trace");
  material_data_list[where+" Trace"] = data;
  data = vector_to_string(values,"Norm");
  material_data_list[where+" Norm"] = data;
}

string VariablePlotter::vector_to_string(vector< int > data) {
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string VariablePlotter::vector_to_string(vector< string > data) {
  string result;
  for(int i = 0; i < (int)data.size(); i++) {
      result+= (data[i] + " ");
    }
  return result;
}

string VariablePlotter::vector_to_string(vector< double > data) {
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string VariablePlotter::vector_to_string(vector< Vector > data, string type) {
  ostringstream ostr;
  if (type == "length") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length() << " ";
    }
  } else if (type == "length2") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length2() << " ";
    }
  } else if (type == "x") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].x() << " ";
    }
  } else if (type == "y") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].y() << " ";
    }
  } else if (type == "z") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].z() << " ";
    }
  }

  return ostr.str();
}

string VariablePlotter::vector_to_string(vector< Matrix3 > data, string type) {
  ostringstream ostr;
  if (type == "Determinant") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Determinant() << " ";
    }
  } else if (type == "Trace") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Trace() << " ";
    }
  } else if (type == "Norm") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Norm() << " ";
    } 
 }

  return ostr.str();
}

string VariablePlotter::currentNode_str() {
  ostringstream ostr;
  ostr << "Level-" << currentNode.level << "-(";
  ostr << currentNode.id.x()  << ",";
  ostr << currentNode.id.y()  << ",";
  ostr << currentNode.id.z() << ")";
  return ostr.str();
}

void VariablePlotter::extract_data(string display_mode, string varname,
				  vector <string> mat_list,
				  vector <string> type_list, string index) {

  // update currentNode with the values in the tcl code
  pick();
  
  // clear the current contents of the ticles's material data list
  TCL::execute(id + " reset_var_val");

  // determine type
  const TypeDescription *td;
  for(int i = 0; i < (int)names.size() ; i++)
    if (names[i] == varname)
      td = types[i];
  
  string name_list("");
  const TypeDescription* subtype = td->getSubType();
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
    cerr << "Graphing a variable of type double\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(currentNode_str()+" "+varname+" "+mat_list[i],data)) {
	// query the value and then cache it
	vector< double > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive->query(values, varname, matl, currentNode.id, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	  return;
	} 
	cerr << "Received data.  Size of data = " << values.size() << endl;
	cache_value(currentNode_str()+" "+varname+" "+mat_list[i],values,data);
      } else {
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Vector:
    cerr << "Graphing a variable of type Vector\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(currentNode_str()+" "+varname+" "+mat_list[i]+" "
		     +type_list[i],data)) {
	// query the value and then cache it
	vector< Vector > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive->query(values, varname, matl, currentNode.id, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	  return;
	} 
	cerr << "Received data.  Size of data = " << values.size() << endl;
	// could integrate this in with the cache_value somehow
	data = vector_to_string(values,type_list[i]);
	cache_value(currentNode_str()+" "+varname+" "+mat_list[i],values);
      } else {
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Matrix3:
    cerr << "Graphing a variable of type Matrix3\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(currentNode_str()+" "+varname+" "+mat_list[i]+" "
		     +type_list[i],data)) {
	// query the value and then cache it
	vector< Matrix3 > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive->query(values, varname, matl, currentNode.id, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	  return;
	} 
	cerr << "Received data.  Size of data = " << values.size() << endl;
	// could integrate this in with the cache_value somehow
	data = vector_to_string(values,type_list[i]);
	cache_value(currentNode_str()+" "+varname+" "+mat_list[i],values);
      }
      else {
	// use cached value that was put into data by is_cached
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Point:
    cerr << "Error trying to graph a Point.  No valid representation for Points for 2d graph.\n";
    break;
  default:
    cerr<<"Unknown var type\n";
    }// else { Tensor,Other}
  TCL::execute(id+" "+display_mode.c_str()+"_data "+index.c_str()+" "
	       +varname.c_str()+" "+currentNode_str().c_str()+" "
	       +name_list.c_str());
  
}



// if a pick event was received extract the id from the picked
void VariablePlotter::pick() {
  reset_vars();
  currentNode.id.x(index_x.get());
  currentNode.id.y(index_y.get());
  currentNode.id.z(index_z.get());
  currentNode.level = index_l.get();
  cerr << "Extracting values for " << currentNode.id << ", level " << currentNode.level << endl;
}

} // End namespace Uintah

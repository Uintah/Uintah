/*
 *  VariablePlotter.cc:  Displays plots for simulation variables
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

#include <Packages/Uintah/Dataflow/Modules/Visualization/VariablePlotter.h>

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Containers/StringUtil.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/VariableCache.h>
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

static string widget_name("VariablePlotter Widget");
 
  DECLARE_MAKER(VariablePlotter)
  
  VariablePlotter::VariablePlotter(const string& name, GuiContext* ctx)
: Module(name, ctx, Filter, "Visualization", "Uintah"),
  var_orientation(ctx->subVar("var_orientation")),
  nl(ctx->subVar("nl")),
  index_x(ctx->subVar("index_x")),
  index_y(ctx->subVar("index_y")),
  index_z(ctx->subVar("index_z")),
  index_l(ctx->subVar("index_l")),
  curr_var(ctx->subVar("curr_var")),
  old_generation(-1), old_timestep(0), grid(NULL)
{

}

  VariablePlotter::VariablePlotter(GuiContext* ctx)
: Module("VariablePlotter", ctx, Filter, "Visualization", "Uintah"),
  var_orientation(ctx->subVar("var_orientation")),
  nl(ctx->subVar("nl")),
  index_x(ctx->subVar("index_x")),
  index_y(ctx->subVar("index_y")),
  index_z(ctx->subVar("index_z")),
  index_l(ctx->subVar("index_l")),
  curr_var(ctx->subVar("curr_var")),
  old_generation(-1), old_timestep(0), grid(NULL)
{

}


VariablePlotter::~VariablePlotter()
{
}

// assigns a grid based on the archive and the timestep to grid
// return true if there was a new grid (either completely or just a new
// timestep), false otherwise.
bool VariablePlotter::getGrid()
{
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"VariablePlotter::getGrid::Didn't get a handle\n";
    grid = NULL;
    return false;
  }

  // access the grid through the handle and dataArchive
  archive = (*(handle.get_rep()))();
  int new_generation = (*(handle.get_rep())).generation;
  bool archive_dirty =  new_generation != old_generation;
  int timestep = (*(handle.get_rep())).timestep();
  if (archive_dirty) {
    old_generation = new_generation;
    vector< int > indices;
    times.clear();
    archive->queryTimesteps( indices, times );
    gui->execute(id + " set_time " +
		 VariableCache::vector_to_string(indices).c_str());
    // set old_timestep to something that will cause a new grid
    // to be queried.
    old_timestep = -1;
    // clean out the cached information if the grid has changed
    material_data_list.clear();
  }
  if (timestep != old_timestep) {
    time = times[timestep];
    grid = archive->queryGrid(time);
    old_timestep = timestep;
    return true;
  }
  return false;
}

// Returns 0 on sucess, 1 otherwise.
int VariablePlotter::add_type(string &type_list,const TypeDescription *subtype)
{
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
  case TypeDescription::int_type:
    type_list += " scaler";
    break;
  case TypeDescription::Vector:
    type_list += " vector";
    break;
  case TypeDescription::Matrix3:
    type_list += " matrix3";
    break;
  // These need to be added as they get implemented
  case TypeDescription::bool_type:
    error("Error in VariablePlotter::add_type(): Boolean subtype is not implemented.");
    return 1;
  case TypeDescription::short_int_type:
    error("Error in VariablePlotter::add_type(): Short Int subtype is not implemented.");
    return 1;
  case TypeDescription::long_type:
    error("Error in VariablePlotter::add_type(): Long subtype is not implemented.");
    return 1;
  case TypeDescription::long64_type:
    error("Error in VariablePlotter::add_type(): Long64 subtype is not implemented.");
    return 1;
  default:
    error("Error in VariablePlotter::add_type(): Unknown subtype not implemented.");
    error("Please tell Vis Team about this, and prepare the net you used as well as a copy of the data.");
    return 1;
  }
  return 0;
}  

void VariablePlotter::setVars(GridP grid) {
  string varNames("");
  string type_list("");
  const Patch* patch = *(grid->getLevel(0)->patchesBegin());

  cerr << "Calling clearMat_list\n";
  gui->execute(id + " clearMat_list ");
  
  for(int i = 0; i< (int)names.size(); i++) {
    switch (types[i]->getType()) {
    case TypeDescription::NCVariable:
      if (var_orientation.get() == NC_VAR) {
	if (!add_type(type_list,types[i]->getSubType())) {
	  // Only add this stuff to the gui if the variable was added
	  // successfully
	  varNames += " ";
	  varNames += names[i];
	  cerr << "Calling appendMat_list for "<<names[i]<<"\n";
	  gui->execute(id + " appendMat_list " + archive->queryMaterials(names[i], patch, time).expandedString().c_str());
	} else {
	  error("Variable " +  names[i] + " was not added, because its subtype is not supported.");
	}
      }
      break;
    case TypeDescription::CCVariable:
      if (var_orientation.get() == CC_VAR) {
	if (!add_type(type_list,types[i]->getSubType())) {
	  // Only add this stuff to the gui if the variable was added
	  // successfully
	  varNames += " ";
	  varNames += names[i];
	  cerr << "Calling appendMat_list for "<<names[i]<<"\n";
	  gui->execute(id + " appendMat_list " + archive->queryMaterials(names[i], patch, time).expandedString().c_str());
	} else {
	  error("Variable " +  names[i] + " was not added, because its subtype is not supported.");
	}
      }
      break;
    default:
      cerr << "VariablePlotter::setVars: Warning!  Ignoring unknown type for variable "<<names[i]<<".\n";
      break;
    }

    
  }

  cerr << "varNames = " << varNames << endl;
  gui->execute(id + " setVar_list " + varNames.c_str());
  gui->execute(id + " setType_list " + type_list.c_str());  
}

void VariablePlotter::initialize_ports() {
  // Create the input port
  in= (ArchiveIPort *) get_iport("Data Archive");
}

int VariablePlotter::initialize_grid() {
  // Get the handle on the grid and the number of levels
  bool new_grid = getGrid();
  if(!grid)
    return 2;
  numLevels = grid->numLevels();

  // setup the tickle stuff
  if (new_grid) {
    nl.set(numLevels);
    names.clear();
    types.clear();
    archive->queryVariables(names, types);
  }
  setVars(grid);

  if (new_grid)
    return 1;
  else
    return 0;
}

void VariablePlotter::update_tcl_window() {
  string visible;
  gui->eval(id + " isVisible", visible);
  if ( visible == "1") {
    gui->execute(id + " destroyFrames");
    gui->execute(id + " build");

    gui->execute("update idletasks");
    reset_vars();
  }
}

void VariablePlotter::execute()
{
  cerr << "VariablePlotter::execute:start\n";
  initialize_ports();
  
  if (initialize_grid() == 2)
    return;
  
  update_tcl_window();
  
  cerr << "VariablePlotter::execute:end\n";
}

void VariablePlotter::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  else if(args[1] == "update_sn") {
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
    cerr << "Graphing " << varname << " with materials: " <<
      VariableCache::vector_to_string(mat_list) << endl;
    extract_data(displaymode,varname,mat_list,type_list,index);
  }
  else {
    Module::tcl_command(args, userdata);
  }
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
  gui->execute(id + " reset_var_val");

  // determine type
  const TypeDescription *td;
  for(int i = 0; i < (int)names.size() ; i++)
    if (names[i] == varname)
      td = types[i];
  
  string name_list("");
  // Key to use for the VariableCache.  This can be used as is unless you
  // are accessing a Vector or Matrix3.  Then you will have to add a suffix
  // for the scalar value associated with what you want.  See VariableCache.h
  // for details.
  const TypeDescription* subtype = td->getSubType();
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
    cerr << "Graphing a variable of type double\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      string cache_key(currentNode_str()+" "+varname+" "+mat_list[i]);
      if (!material_data_list.get_cached(cache_key,data)) {
	cerr << "Cache miss.  Querying the data archive\n";
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
	material_data_list.cache_value(cache_key, values, data);
      } else {
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::int_type:
    cerr << "Graphing a variable of type int\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      string cache_key(currentNode_str()+" "+varname+" "+mat_list[i]);
      if (!material_data_list.get_cached(cache_key,data)) {
	cerr << "Cache miss.  Querying the data archive\n";
	// query the value and then cache it
	vector< int > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive->query(values, varname, matl, currentNode.id, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	  return;
	} 
	cerr << "Received data.  Size of data = " << values.size() << endl;
	material_data_list.cache_value(cache_key, values, data);
      } else {
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Vector:
    cerr << "Graphing a variable of type Vector\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      string cache_key(currentNode_str()+" "+varname+" "+mat_list[i]);
      // The suffix to get things like, length, lenght2 and what not.
      string type_suffix(" "+type_list[i]);
      if (!material_data_list.get_cached(cache_key + type_suffix,data)) {
	cerr << "Cache miss.  Querying the data archive\n";
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
	material_data_list.cache_value(cache_key, values);
	material_data_list.get_cached(cache_key + type_suffix, data);
      } else {
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Matrix3:
    cerr << "Graphing a variable of type Matrix3\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      string cache_key(currentNode_str()+" "+varname+" "+mat_list[i]);
      // The suffix to get things like, Determinant and what not.
      string type_suffix(" "+type_list[i]);
      if (!material_data_list.get_cached(cache_key + type_suffix, data)) {
	cerr << "Cache miss.  Querying the data archive\n";
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
	material_data_list.cache_value(cache_key, values);
	material_data_list.get_cached(cache_key + type_suffix, data);
      }
      else {
	// use cached value that was put into data by is_cached
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Point:
    cerr << "Error trying to graph a Point.  No valid representation for Points for 2d graph.\n";
    break;
  default:
    cerr<<"Unknown var type\n";
    }// else { Tensor,Other}
  gui->execute(id+" "+display_mode.c_str()+"_data "+index.c_str()+" "
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

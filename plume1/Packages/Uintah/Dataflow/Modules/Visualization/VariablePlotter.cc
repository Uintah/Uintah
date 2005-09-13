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
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/VariableCache.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h> // Includ after Patch.h
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h> // Includ after Patch.h
//#include <Packages/Uintah/Core/Grid/FaceIterator.h> // Includ after Patch.h
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <vector>
#include <sstream>
#include <iostream>
//#include <string>


using namespace SCIRun;
using namespace Uintah;
using namespace std;


// should match the values in the tcl code
#define NC_VAR 0
#define CC_VAR 1

static string widget_name("VariablePlotter Widget");
 
DECLARE_MAKER(VariablePlotter)
  
VariablePlotter::VariablePlotter(const string& name, GuiContext* ctx) :
  Module(name, ctx, Filter, "Visualization", "Uintah"),
  var_orientation_(ctx->subVar("var_orientation")),
  nl_(ctx->subVar("nl")),
  index_x_(ctx->subVar("index_x")),
  index_y_(ctx->subVar("index_y")),
  index_z_(ctx->subVar("index_z")),
  index_l_(ctx->subVar("index_l")),
  curr_var_(ctx->subVar("curr_var")),
  old_generation_(-1), old_timestep_(0), grid_(NULL)
{
}

VariablePlotter::VariablePlotter(GuiContext* ctx) :
  Module("VariablePlotter", ctx, Filter, "Visualization", "Uintah"),
  var_orientation_(ctx->subVar("var_orientation")),
  nl_(ctx->subVar("nl")),
  index_x_(ctx->subVar("index_x")),
  index_y_(ctx->subVar("index_y")),
  index_z_(ctx->subVar("index_z")),
  index_l_(ctx->subVar("index_l")),
  curr_var_(ctx->subVar("curr_var")),
  old_generation_(-1), old_timestep_(0), grid_(NULL)
{
}


VariablePlotter::~VariablePlotter()
{
}

// assigns a grid based on the archive and the timestep to grid
// return true if there was a new grid (either completely or just a new
// timestep), false otherwise.
bool
VariablePlotter::getGrid()
{
  ArchiveHandle handle;
  if (!(in_->get(handle) && handle.get_rep())) {
    warning("VariablePlotter::getGrid - No input data\n");
    grid_ = NULL;
    return false;
  }

  // access the grid through the handle and dataArchive
  archive_ = handle->getDataArchive();
  int new_generation = handle->generation;
  bool archive_dirty =  new_generation != old_generation_;
  int timestep = handle->timestep();
  if (archive_dirty) {
    old_generation_ = new_generation;
    vector< int > indices;
    times_.clear();
    archive_->queryTimesteps( indices, times_ );
    gui->execute(id + " set_time " +
                 VariableCache::vector_to_string(times_).c_str());
    // set old_timestep to something that will cause a new grid
    // to be queried.
    old_timestep_ = -1;
    // clean out the cached information if the grid has changed
    material_data_list_.clear();
  }
  if (timestep != old_timestep_) {
    time_ = times_[timestep];
    grid_ = archive_->queryGrid(time_);
    old_timestep_ = timestep;
    return true;
  }
  return false;
}

// Returns 0 on sucess, 1 otherwise.
int
VariablePlotter::add_type(string &type_list,const TypeDescription *subtype)
{
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
  case TypeDescription::float_type:
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

void
VariablePlotter::setVars(GridP grid) 
{
  string varNames("");
  string type_list("");
  const Patch* patch = *(grid->getLevel(0)->patchesBegin());

  cerr << "Calling clearMat_list\n";
  gui->execute(id + " clearMat_list ");
  
  for(int i = 0; i< (int)names_.size(); i++) {
    switch (types_[i]->getType()) {
    case TypeDescription::NCVariable:
      if (var_orientation_.get() == NC_VAR) {
        if (!add_type(type_list,types_[i]->getSubType())) {
          // Only add this stuff to the gui if the variable was added
          // successfully
          varNames += " ";
          varNames += names_[i];
          cerr << "Calling appendMat_list for "<<names_[i]<<"\n";
          gui->execute(id + " appendMat_list " + archive_->queryMaterials(names_[i], patch, time_).expandedString().c_str());
        } else {
          error("Variable " +  names_[i] + " was not added, because its subtype is not supported.");
        }
      }
      break;
    case TypeDescription::CCVariable:
      if (var_orientation_.get() == CC_VAR) {
        if (!add_type(type_list,types_[i]->getSubType())) {
          // Only add this stuff to the gui if the variable was added
          // successfully
          varNames += " ";
          varNames += names_[i];
          cerr << "Calling appendMat_list for "<<names_[i]<<"\n";
          gui->execute(id + " appendMat_list " + archive_->queryMaterials(names_[i], patch, time_).expandedString().c_str());
        } else {
          error("Variable " +  names_[i] + " was not added, because its subtype is not supported.");
        }
      }
      break;
    default:
      cerr << "VariablePlotter::setVars: Warning!  Ignoring unknown type for variable "<<names_[i]<<".\n";
      break;
    }

    
  }

  cerr << "varNames = " << varNames << endl;
  gui->execute(id + " setVar_list " + varNames.c_str());
  gui->execute(id + " setType_list " + type_list.c_str());  
}

void
VariablePlotter::initialize_ports() 
{
  // Create the input port
  in_ = (ArchiveIPort *) get_iport("Data Archive");
}

int
VariablePlotter::initialize_grid() 
{
  // Get the handle on the grid and the number of levels
  bool new_grid = getGrid();
  if(!grid_)
    return 2;
  numLevels_ = grid_->numLevels();

  // setup the tickle stuff
  if (new_grid) {
    cerr << "VariablePlotter::initialize_grid(): new_grid is true\n";
    nl_.set(numLevels_);
    names_.clear();
    types_.clear();
    archive_->queryVariables(names_, types_);
  } else {
    cerr << "VariablePlotter::initialize_grid(): new_grid is false\n";
  }    
  setVars(grid_);

  if (new_grid)
    return 1;
  else
    return 0;
}

void
VariablePlotter::update_tcl_window() 
{
  string visible;
  gui->eval(id + " isVisible", visible);
  if ( visible == "1") {
    gui->execute(id + " destroyFrames");
    gui->execute(id + " build");

    gui->execute("update idletasks");
    reset_vars();
  }
}

void
VariablePlotter::execute()
{
  //cerr << "VariablePlotter::execute:start\n";
  initialize_ports();
  
  switch (initialize_grid()) {
  case 0:
    // Grid didn't change, so don't update gui
    update_tcl_window();
    break;
  case 1:
    // Grid changed so update the gui
    update_tcl_window();
    break;
  case 2:
    // Didn't get a grid handle
    return;
  }
  //cerr << "VariablePlotter::execute:end\n";
}

void
VariablePlotter::tcl_command(GuiArgs& args, void* userdata)
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

string
VariablePlotter::currentNode_str() 
{
  ostringstream ostr;
  ostr << "Level-" << currentNode_.level_ << "-(";
  ostr << currentNode_.id_.x()  << ",";
  ostr << currentNode_.id_.y()  << ",";
  ostr << currentNode_.id_.z() << ")";
  return ostr.str();
}

void
VariablePlotter::extract_data( string display_mode, string varname,
                               vector <string> mat_list,
                               vector <string> type_list, string index) 
{

  // update currentNode with the values in the tcl code
  pick();
  
  // clear the current contents of the ticles's material data list
  gui->execute(id + " reset_var_val");

  // determine type
  const TypeDescription *td = NULL;
  for(int i = 0; i < (int)names_.size() ; i++) {
    if (names_[i] == varname) {
      td = types_[i];
    }
  }
  
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
      if (!material_data_list_.get_cached(cache_key,data)) {
        cerr << "Cache miss.  Querying the data archive\n";
        // query the value and then cache it
        vector< double > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive_->query(values, varname, matl, currentNode_.id_, times_[0], 
                         times_[times_.size()-1], currentNode_.level_);
        } catch (const VariableNotFoundInGrid& exception) {
          cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
          return;
        } 
        cerr << "Received data.  Size of data = " << values.size() << endl;
        material_data_list_.cache_value(cache_key, values, data);
      } else {
        cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::float_type:
    cerr << "Graphing a variable of type float\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      string cache_key(currentNode_str()+" "+varname+" "+mat_list[i]);
      if (!material_data_list_.get_cached(cache_key,data)) {
        cerr << "Cache miss.  Querying the data archive\n";
        // query the value and then cache it
        vector< float > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive_->query(values, varname, matl, currentNode_.id_, times_[0], times_[times_.size()-1],
                          currentNode_.level_);
        } catch (const VariableNotFoundInGrid& exception) {
          cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
          return;
        } 
        cerr << "Received data.  Size of data = " << values.size() << endl;
        material_data_list_.cache_value(cache_key, values, data);
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
      if (!material_data_list_.get_cached(cache_key,data)) {
        cerr << "Cache miss.  Querying the data archive\n";
        // query the value and then cache it
        vector< int > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive_->query(values, varname, matl, currentNode_.id_, times_[0], times_[times_.size()-1],
                          currentNode_.level_);
        } catch (const VariableNotFoundInGrid& exception) {
          cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
          return;
        } 
        cerr << "Received data.  Size of data = " << values.size() << endl;
        material_data_list_.cache_value(cache_key, values, data);
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
      if (!material_data_list_.get_cached(cache_key + type_suffix,data)) {
        cerr << "Cache miss.  Querying the data archive\n";
        // query the value and then cache it
        vector< Vector > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive_->query(values, varname, matl, currentNode_.id_, times_[0], times_[times_.size()-1],
                          currentNode_.level_);
        } catch (const VariableNotFoundInGrid& exception) {
          cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
          return;
        } 
        cerr << "Received data.  Size of data = " << values.size() << endl;
        material_data_list_.cache_value(cache_key, values);
        material_data_list_.get_cached(cache_key + type_suffix, data);
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
      if (!material_data_list_.get_cached(cache_key + type_suffix, data)) {
        cerr << "Cache miss.  Querying the data archive\n";
        // query the value and then cache it
        vector< Matrix3 > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive_->query(values, varname, matl, currentNode_.id_, times_[0], times_[times_.size()-1],
                          currentNode_.level_);
        } catch (const VariableNotFoundInGrid& exception) {
          cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
          return;
        } 
        cerr << "Received data.  Size of data = " << values.size() << endl;
        material_data_list_.cache_value(cache_key, values);
        material_data_list_.get_cached(cache_key + type_suffix, data);
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
  
} // end extract_data()


// if a pick event was received extract the id from the picked
void
VariablePlotter::pick()
{
  reset_vars();
  currentNode_.id_.x(index_x_.get());
  currentNode_.id_.y(index_y_.get());
  currentNode_.id_.z(index_z_.get());
  currentNode_.level_ = index_l_.get();
  cerr << "Extracting values for " << currentNode_.id_ << ", level " << currentNode_.level_ << endl;
}



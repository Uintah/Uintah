/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  NrrdFieldConverter.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Packages/Fusion/Dataflow/Modules/Fields/VULCANConverter.h>

namespace Fusion {

using namespace SCIRun;

class VULCANConverter : public Module {
public:
  VULCANConverter(GuiContext*);

  virtual ~VULCANConverter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  enum { NONE = 0, MESH = 1, SCALAR = 2, REALSPACE = 4, CONNECTION = 8 };
  enum { ZR = 0, PHI = 1, LIST = 2 };

  GuiString gui_datasets_;
  GuiInt gui_nmodes_;
  vector< GuiInt* > gui_modes_;

  int nmodes_;
  vector< int > modes_;

  unsigned int conversion_;

  GuiInt gui_allow_unrolling_;
  GuiInt gui_unrolling_;

  vector< int > mesh_;
  vector< int > data_;

  unsigned int nHandles_;

  NrrdDataHandle nrrd_output_handle_;

  bool execute_error_;
};


DECLARE_MAKER(VULCANConverter)
VULCANConverter::VULCANConverter(GuiContext* context)
  : Module("VULCANConverter", context, Source, "Fields", "Fusion"),
    gui_datasets_(context->subVar("datasets")),
    gui_nmodes_(context->subVar("nmodes")),
    conversion_(NONE),
    gui_allow_unrolling_(context->subVar("allowUnrolling")),
    gui_unrolling_(context->subVar("unrolling")),
    nrrd_output_handle_(0),
    execute_error_(false)
{
}

VULCANConverter::~VULCANConverter(){
}

void
VULCANConverter::execute(){

  vector< NrrdDataHandle > nrrd_input_handles;
  NrrdDataHandle nrrd_input_handle;

  if( !get_dynamic_input_handles( "Input Nrrd", nrrd_input_handles, true ) )
    return;

  string datasetsStr;

  string nrrdName;
  string property;

  for( unsigned int ic=0; ic<nrrd_input_handles.size(); ic++ ) {

    // Get the source of the nrrd being worked on.
    if( !nrrd_input_handles[ic]->get_property( "Source", property ) ||
	property == "Unknown" ) {
      error( "Can not find the source of the nrrd or it is unknown." );
      return;
    }

    nrrd_input_handles[ic]->get_property( "Name", nrrdName );

    // Save the name of the dataset.
    if( nrrd_input_handles.size() == 1 )
      datasetsStr.append( nrrdName );
    else
      datasetsStr.append( "{" + nrrdName + "} " );
  }

  if( datasetsStr != gui_datasets_.get() ) {
    // Update the dataset names and dims in the GUI.
    ostringstream str;
    str << get_id() << " set_names " << " {" << datasetsStr << "}";
    
    get_gui()->execute(str.str().c_str());
  }

  if( nrrd_input_handles.size() != 1 && nrrd_input_handles.size() != 2 &&
      nrrd_input_handles.size() != 3 &&
      nrrd_input_handles.size() != 4 && nrrd_input_handles.size() != 8 ){
    error( "Not enough or too many handles or representations" );
    return;
  }

  if( nHandles_ != nrrd_input_handles.size() )
    inputs_changed_ = true;

  // If data change, update the GUI the field if needed.
  if( inputs_changed_ ) {

    nHandles_ = nrrd_input_handles.size();

    conversion_ = NONE;

    remark( "Found new data ... updating." );

    mesh_.resize(4);
    mesh_[0] = mesh_[1] = mesh_[2] = mesh_[3] = -1;

    // Get each of the dataset names for the GUI.
    for( unsigned int ic=0; ic<nrrd_input_handles.size(); ic++ ) {

      NrrdDataHandle nHandle = nrrd_input_handles[ic];

      // Get the name of the nrrd being worked on.
      nHandle->get_property( "Name", nrrdName );

      if( nHandle->get_property( "Topology", property ) ) {

	// Structured mesh.
	if( property.find( "Structured" ) != string::npos ||
	    property.find( "Unstructured" ) != string::npos ) {

	  if( nHandle->get_property( "Coordinate System", property ) ) {

	    // Special Case - VULCAN data which has multiple components.
	    if( property.find("Cylindrical - VULCAN") != string::npos ) {

	      // Sort the components.
	      if( nrrdName.find( "ZR:Scalar" ) != string::npos &&
		  nHandle->nrrd_->dim == 2 ) {
		conversion_ = MESH;
		mesh_[ZR] = ic;
	      } else if( nrrdName.find( "ZR:Vector" ) != string::npos &&
			 nHandle->nrrd_->dim == 2 ) {
		conversion_ = MESH;
		mesh_[ZR] = ic;
	      } else if( nrrdName.find( "PHI:Scalar" ) != string::npos && 
			 nHandle->nrrd_->dim == 1 ) {
		mesh_[PHI] = ic;
	      } else {
		error( nrrdName + " is unknown VULCAN mesh data." );
		execute_error_ = true;
		return;
	      }
	    } else {
	      error( nrrdName + " " + property + " is an unsupported coordinate system." );
	      execute_error_ = true;
	      return;
	    }
	  } else if( nHandle->get_property( "Cell Type", property ) ) {
	    conversion_ = CONNECTION;
	    mesh_[LIST] = ic;
	  } else {
	    error( nrrdName + "No coordinate system or cell type found." );
	    execute_error_ = true;
	    return;
	  }
	} else {
	  error( nrrdName + " " + property + " is an unsupported topology." );
	  execute_error_ = true;
	  return;
	}
      } else if( nHandle->get_property( "DataSpace", property ) ) {

	if( property.find( "REALSPACE" ) != string::npos ) {

	  if( data_.size() == 0 ) {
	    data_.resize(1);
	    data_[0] = -1;
	  }

	  if( nrrdName.find( "ZR:Scalar" ) != string::npos && 
	      nHandle->nrrd_->dim == 2 ) {
	    conversion_ = REALSPACE;
	    data_[ZR] = ic; 
	  } else if( nrrdName.find( "ZR:Vector" ) != string::npos && 
	      nHandle->nrrd_->dim == 2 ) {
	    conversion_ = REALSPACE;
	    data_[ZR] = ic; 
	  } else if( nrrdName.find( ":Scalar" ) != string::npos && 
		     nHandle->nrrd_->dim == 1 ) {
	    conversion_ = SCALAR;
	    data_[0] = ic;
	  } else {
	    error( nrrdName + " is unknown VULCAN node data." );
	    execute_error_ = true;
	    return;
	  }
	} else {	
	  error( nrrdName + " " + property + " Unsupported Data Space." );
	  execute_error_ = true;
	  return;
	}
      } else {
	error( nrrdName + " No DataSpace property." );
	execute_error_ = true;
	return;
      }
    }
  }

  unsigned int i = 0;
  while( i<data_.size() && data_[i] != -1 )
    i++;
  
  if( conversion_ & MESH ) {
    if( mesh_[PHI] == -1 || mesh_[ZR] == -1 ) {
      error( "Not enough data for the mesh conversion." );
      execute_error_ = true;
      return;
    }
  } else if( conversion_ & CONNECTION ) {
    if( mesh_[PHI] == -1 || mesh_[ZR] == -1 || mesh_[LIST] == -1 ) {
      error( "Not enough data for the connection conversion." );
      execute_error_ = true;
      return;
    }
  } else if ( conversion_ & REALSPACE ) {
    if( mesh_[PHI] == -1 || data_[ZR] == -1 ) {

      error( "Not enough data for the realspace conversion." );

      execute_error_ = true;
      return;
    }
  } else if ( conversion_ & SCALAR ) {
    if( mesh_[PHI] == -1 || mesh_[ZR] == -1 || mesh_[LIST] == -1 || data_[ZR] == -1 ) {

      error( "Not enough data for the scalar conversion." );

      execute_error_ = true;
      return;
    }
  }

  if( (int) (conversion_ & MESH) != gui_allow_unrolling_.get() ) {
    ostringstream str;
    str << get_id() << " set_unrolling " << (conversion_ & MESH);
    
    get_gui()->execute(str.str().c_str());

    if( conversion_ & MESH ) {
      warning( "Select the mesh rolling for the calculation" );
      execute_error_ = true; // Not really an error but it so it will execute.
      return;
    }
  }

  nmodes_ = 0;

  int nmodes = gui_modes_.size();

  // Remove the GUI entries that are not needed.
  for( int ic=nmodes-1; ic>nmodes_; ic-- ) {
    delete( gui_modes_[ic] );
    gui_modes_.pop_back();
    modes_.pop_back();
  }

  if( nmodes_ > 0 ) {
    // Add new GUI entries that are needed.
    for( int ic=nmodes; ic<=nmodes_; ic++ ) {
      char idx[24];
      
      sprintf( idx, "mode-%d", ic );
      gui_modes_.push_back(new GuiInt(get_ctx()->subVar(idx)) );
      
      modes_.push_back(0);
    }
  }

  if( gui_nmodes_.get() != nmodes_ ) {

    // Update the modes in the GUI
    ostringstream str;
    str << get_id() << " set_modes " << nmodes_ << " 1";

    get_gui()->execute(str.str().c_str());
    
  }


  bool updateMode = false;

  if( nmodes_ > 0 ) {
    bool haveMode = false;

    for( int ic=0; ic<=nmodes_; ic++ ) {
      gui_modes_[ic]->reset();
      if( modes_[ic] != gui_modes_[ic]->get() ) {
	modes_[ic] = gui_modes_[ic]->get();
	updateMode = true;
      }

      if( !haveMode ) haveMode = (modes_[ic] ? true : false );
    }

    if( !haveMode ) {
      warning( "Select the mode for the calculation" );
      execute_error_ = true; // Not really an error but it so it will execute.
      return;
    }
  }

  // If no data or data change, recreate the field.
  if( inputs_changed_ ||

      !nrrd_output_handle_.get_rep() ||

      gui_unrolling_.changed( true ) ||

      updateMode ||

      execute_error_ ) {
    
    update_state( Executing );

    execute_error_ = false;

    string convertStr;
    unsigned int ntype;

    if( conversion_ & MESH ) {
      ntype = nrrd_input_handles[mesh_[PHI]]->nrrd_->type;

      nrrd_input_handles[mesh_[PHI]]->get_property( "Coordinate System", property );

      if( property.find("Cylindrical - VULCAN") != string::npos ) {
	modes_.resize(1);
	modes_[0] = gui_unrolling_.get();
      }

      convertStr = "Mesh";

    } else if( conversion_ & CONNECTION ) {
      ntype = nrrd_input_handles[mesh_[LIST]]->nrrd_->type;

      nrrd_input_handles[mesh_[PHI]]->get_property( "Coordinate System", property );

      if( property.find("Cylindrical - VULCAN") != string::npos ) {
	modes_.resize(1);
	modes_[0] = gui_unrolling_.get();
      }

      convertStr = "Connection";

    } else if( conversion_ & SCALAR ) {
      ntype = nrrd_input_handles[data_[0]]->nrrd_->type;

      convertStr = "Scalar";

     } else if( conversion_ & REALSPACE ) {
      ntype = nrrd_input_handles[mesh_[PHI]]->nrrd_->type;

      convertStr = "RealSpace";
    }

    if( conversion_ ) {
      remark( "Converting the " + convertStr );
    
      CompileInfoHandle ci_mesh =
	VULCANConverterAlgo::get_compile_info( convertStr, ntype );
    
      Handle<VULCANConverterAlgo> algo_mesh;
    

      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) {
	error( "No Module" );
	execute_error_ = true;
	return;
      }

      nrrd_output_handle_ =
	algo_mesh->execute( nrrd_input_handles, mesh_, data_, modes_ );
    } else {
      error( "Nothing to convert." );
      execute_error_ = true;
      return;
    }

    if( conversion_ & MESH )
      modes_.clear();
  }
  
  // Send the data downstream
  send_output_handle( "Output Nrrd", nrrd_output_handle_, true );
}

void
VULCANConverter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


void get_nrrd_compile_type( const unsigned int type,
			    string & typeStr,
			    string & typeName );

CompileInfoHandle
VULCANConverterAlgo::get_compile_info( const string converter,
				       const unsigned int ntype )
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string base_class_name( "VULCANConverterAlgo");
  const string template_class_name( "VULCAN" + converter + "ConverterAlgoT");

  string nTypeStr, nTypeName;

  get_nrrd_compile_type( ntype, nTypeStr, nTypeName );

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       nTypeName + ".",
                       base_class_name, 
                       template_class_name,
                       nTypeStr );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  return rval;
}

}

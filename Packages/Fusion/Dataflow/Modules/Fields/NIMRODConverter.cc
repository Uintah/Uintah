/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Packages/Fusion/Dataflow/Modules/Fields/NIMRODConverter.h>

#include <sci_defs.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class PSECORESHARE NIMRODConverter : public Module {
public:
  NIMRODConverter(GuiContext*);

  virtual ~NIMRODConverter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  enum { NONE = 0, MESH = 1, REALSPACE = 2, PERTURBED = 4 };

  GuiString datasetsStr_;

  vector< int > mesh_;
  vector< int > data_;

  vector< int > nGenerations_;

  NrrdDataHandle nHandle_;

  bool error_;
};


DECLARE_MAKER(NIMRODConverter)
NIMRODConverter::NIMRODConverter(GuiContext* context)
  : Module("NIMRODConverter", context, Source, "Fields", "Fusion"),
    datasetsStr_(context->subVar("datasets")),

    error_(false)
{
}

NIMRODConverter::~NIMRODConverter(){
}

void
NIMRODConverter::execute(){

  vector< NrrdDataHandle > nHandles;
  NrrdDataHandle nHandle;

  // Assume a range of ports even though four are needed.
  port_range_type range = get_iports("Input Nrrd");

  if (range.first == range.second)
    return;

  port_map_type::iterator pi = range.first;

  while (pi != range.second) {
    NrrdIPort *inrrd_port = (NrrdIPort*) get_iport(pi->second); 

    ++pi;

    if (!inrrd_port) {
      error( "Unable to initialize "+name+"'s iport " );
      return;
    }

    // Save the field handles.
    if (inrrd_port->get(nHandle) && nHandle.get_rep())
      nHandles.push_back( nHandle );
    else if( pi != range.second ) {
      error( "No handle or representation" );
      return;
    }
  }

  if( nHandles.size() != 3 && nHandles.size() != 4 ){
    error( "Not enough or too many handles or representations" );
    return;
  }

  int generation = 0;

  // See if input data has been added or removed.
  if( nGenerations_.size() == 0 ||
      nGenerations_.size() != nHandles.size() )
    generation = nHandles.size();
  else {
    // See if any of the input data has changed.
    for( unsigned int ic=0; ic<nHandles.size() &&
	   ic<nGenerations_.size(); ic++ ) {
      if( nGenerations_[ic] != nHandles[ic]->generation )
	++generation;
    }
  }

  string property;
  unsigned int conversion = NONE;

  // If data change, update the GUI the field if needed.
  if( generation ) {

    nGenerations_.resize( nHandles.size() );
    mesh_.resize(4);
    mesh_[0] = mesh_[1] = mesh_[2] = mesh_[3] = -1;

    data_.resize(3);
    data_[0] = data_[1] = data_[2] = -1;

    for( unsigned int ic=0; ic++; ic<nHandles.size() )
      nGenerations_[ic] = nHandles[ic]->generation;

    string datasetsStr;

    vector< string > datasets;
    unsigned int modes = 0;

    // Get each of the dataset names for the GUI.
    for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {

      nHandle = nHandles[ic];

      vector< string > dataset;

      int tuples = nHandle->get_tuple_axis_size();

      if( tuples != 1 ) {
	error( "Too many tuples listed in the tuple axis." );
	error_ = true;
	return;
      }

      nHandle->get_tuple_indecies(dataset);

      // Do not allow joined Nrrds
      if( dataset.size() != 1 ) {
	error( "Too many sets listed in the tuple axis." );
	error_ = true;
	return;
      }

      // Save the name of the dataset.
      if( nHandles.size() == 1 )
	datasetsStr.append( dataset[0] );
      else
	datasetsStr.append( "{" + dataset[0] + "} " );
      
      if( nHandle->get_property( "Topology", property ) ) {

	// Structured mesh.
	if( property.find( "Structured" ) != string::npos ) {

	  if( nHandle->get_property( "Coordinate System", property ) ) {

	      // Special Case - NIMROD data which has multiple components.
	    if( property.find("Cylindrical - NIMROD") != string::npos ) {

	      // Sort the components components.
	      if( dataset[0].find( "R:Scalar" ) != string::npos &&
		  nHandle->nrrd->dim == 3 ) {
		conversion = MESH;
		mesh_[0] = ic;
	      } else if( dataset[0].find( "Z:Scalar" ) != string::npos && 
		       nHandle->nrrd->dim == 3 ) {
		conversion = MESH;
		mesh_[1] = ic; 
	      } else if( dataset[0].find( "PHI:Scalar" ) != string::npos && 
		       nHandle->nrrd->dim == 2 ) {
		mesh_[2] = ic;
	      } else if( dataset[0].find( "K:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 2 ) {
		conversion = PERTURBED;
		mesh_[2] = ic;
		modes = nHandle->nrrd->axis[1].size;
	      } else {
		error( dataset[0] + " is unknown NIMROD mesh data." );
		error_ = true;
		return;
	      }
	    } else {
	      error( property + " is an unsupported coordinate system." );
	      error_ = true;
	      return;
	    }
	  } else {
	    error( "No coordinate system found." );
	    error_ = true;
	    return;
	  }
	}
      } else if( nHandle->get_property( "DataSpace", property ) ) {

	if( property.find( "REALSPACE" ) != string::npos ) {

	  data_.resize(3);

	  if( dataset[0].find( "R:Scalar" ) != string::npos &&
	      nHandle->nrrd->dim == 4 ) {
	    conversion = REALSPACE;
	    data_[0] = ic;
	  } else if( dataset[0].find( "Z:Scalar" ) != string::npos && 
		     nHandle->nrrd->dim == 4 ) {
	    conversion = REALSPACE;
	    data_[1] = ic; 
	  } else if( dataset[0].find( "PHI:Scalar" ) != string::npos && 
		     nHandle->nrrd->dim == 4 ) {
	    conversion = REALSPACE;
	    data_[2] = ic;
	  } else {
	    error( dataset[0] + " is unknown NIMROD node data." );
	    error_ = true;
	    return;
	  }
	} else if( property.find( "PERTURBED.REAL" ) != string::npos ) {
	  conversion = PERTURBED;
	  data_[1] = ic;

	} else if( property.find( "PERTURBED.IMAG" ) != string::npos ) {
	  conversion = PERTURBED;
	  data_[2] = ic;

	} else {
	  error( property + " Unsupported Data Space." );
	  error_ = true;
	  return;
	}
      } else {	
	error( dataset[0] + " Bad data input." );
	error_ = true;
	return;
      }
    }

    if( datasetsStr != datasetsStr_.get() ) {
      // Update the dataset names and dims in the GUI.
      ostringstream str;
      str << id << " set_names " << modes << " {" << datasetsStr << "}";
      
      gui->execute(str.str().c_str());
    }
  }

  if( conversion & MESH ) {
    if( mesh_[0] == -1 || mesh_[1] == -1 || mesh_[2] == -1 ) {
      error( "Not enough mesh data for the mesh conversion." );
      error_ = true;
      return;
    }
  } else if ( conversion & REALSPACE ) {
    if( mesh_[2] == -1 || data_[0] == -1 || data_[1] == -1 || data_[2] == -1 ) {
      error( "Not enough mesh data for the real conversion." );
      error_ = true;
      return;
    }
  } else if ( conversion & PERTURBED ) {
    if( mesh_[3] == -1 || data_[1] == -1 || data_[2] == -1 ) {
      error( "Not enough mesh data for the perturbed conversion." );
      error( "No mesh present." );
      error_ = true;
      return;
    }
  }

  // If no data or data change, recreate the field.
  if( error_ ||
      !nHandle_.get_rep() ||
      generation ) {
    
    error_ = false;

    int idim, jdim, kdim;

    string convertStr;

    if( conversion & MESH ) {
      if( property.find("Cylindrical - NIMROD") != string::npos ) {
	if( nHandles[mesh_[0]]->nrrd->axis[1].size != 
	    nHandles[mesh_[1]]->nrrd->axis[1].size ||
	    nHandles[mesh_[0]]->nrrd->axis[2].size != 
	    nHandles[mesh_[1]]->nrrd->axis[2].size ) {
	  error( "Mesh dimension mismatch." );
	  error_ = true;
	  return;
	}

	jdim = nHandles[mesh_[0]]->nrrd->axis[1].size; // Radial
	kdim = nHandles[mesh_[0]]->nrrd->axis[2].size; // Theta
	idim = nHandles[mesh_[2]]->nrrd->axis[1].size; // Phi
      }

      convertStr = "Mesh";

    } else if( conversion & REALSPACE ) {
      if( property.find("Cylindrical - NIMROD") != string::npos ) {
	jdim = nHandles[data_[0]]->nrrd->axis[1].size; // Radial
	kdim = nHandles[data_[0]]->nrrd->axis[2].size; // Theta
	idim = nHandles[mesh_[2]]->nrrd->axis[1].size; // Phi

	for( unsigned int ic=0; ic<data_.size(); ic++ ) {
	  if( nHandles[data_[0]]->nrrd->axis[1].size != jdim ||
	      nHandles[data_[0]]->nrrd->axis[2].size != kdim ||
	      nHandles[data_[0]]->nrrd->axis[3].size != idim ) {
	    error( "Mesh dimension mismatch." );
	    error_ = true;
	    return;
	  }
	}
      }

      convertStr = "RealSpace";
    }

    remark( "Converting the " + convertStr );
    
    CompileInfoHandle ci_mesh =
      NIMRODConverterAlgo::get_compile_info( convertStr,
					     nHandles[mesh_[2]]->nrrd->type);
    
    Handle<NIMRODConverterAlgo> algo_mesh;
    
    if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;
      
    nHandle_ = algo_mesh->execute( nHandles, mesh_, data_, idim, jdim, kdim );
  }
  
  // Get a handle to the output field port.
  if( nHandle_.get_rep() ) {
    NrrdOPort *ofield_port = (NrrdOPort *) get_oport("Output Nrrd");
    
    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( nHandle_ );
  }
}

void
NIMRODConverter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


void get_nrrd_type( const unsigned int type,
		    string & typeStr,
		    string & typeName );

CompileInfoHandle
NIMRODConverterAlgo::get_compile_info( const string converter,
				       const unsigned int ntype )
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string base_class_name( "NIMRODConverterAlgo");
  const string template_class_name( "NIMROD" + converter + "ConverterAlgoT");

  string nTypeStr, nTypeName;

  get_nrrd_type( ntype, nTypeStr, nTypeName );

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

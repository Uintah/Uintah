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
  enum { NONE = 0, MESH = 1, SCALAR = 2, REALSPACE = 4, PERTURBED = 8 };
  enum { R = 0, Z = 1, PHI = 2, K = 3 };

  GuiString datasetsStr_;
  GuiInt nModes_;
  GuiInt iMode_;

  int mode_;

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
    nModes_(context->subVar("nmodes")),
    iMode_(context->subVar("mode")),

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
    if (inrrd_port->get(nHandle) && nHandle.get_rep()) {
      unsigned int tuples = nHandle->get_tuple_axis_size();

      // Store only single nrrds
      if( tuples == 1 ) {
	nHandles.push_back( nHandle );
      } else {

	// Multiple nrrds
	vector< string > dataset;
	nHandle->get_tuple_indecies(dataset);
	
	const unsigned int nrrdDim = nHandle->nrrd->dim;
	
	int *min = scinew int[nrrdDim];
	int *max = scinew int[nrrdDim];

	// Keep the same dims except for the tuple axis.
	for( int j=1; j<nHandle->nrrd->dim; j++) {
	  min[j] = 0;
	  max[j] = nHandle->nrrd->axis[j].size-1;
	}

	// Separate via the tupple axis.
	for( unsigned int i=0; i<tuples; i++ ) {

	  Nrrd *nout = nrrdNew();

	  // Translate the tuple index into the real offsets for a tuple axis.
	  int tmin, tmax;
	  if (! nHandle->get_tuple_index_info( i, i, tmin, tmax)) {
	    error("Tuple index out of range");
	    return;
	  }
	  
	  min[0] = tmin;
	  max[0] = tmax;

	  // Crop the tupple axis.
	  if (nrrdCrop(nout, nHandle->nrrd, min, max)) {

	    char *err = biffGetDone(NRRD);
	    error(string("Trouble resampling: ") + err);
	    msgStream_ << "input Nrrd: nHandle->nrrd->dim="<<nHandle->nrrd->dim<<"\n";
	    free(err);
	  }

	  // Form the new nrrd and store.
	  NrrdData *nrrd = scinew NrrdData;
	  nrrd->nrrd = nout;
	  nout->axis[0].label = strdup(dataset[i].c_str());

	  NrrdDataHandle handle = NrrdDataHandle(nrrd);

	  // Copy the properties.
	  *((PropertyManager *) handle.get_rep()) =
	    *((PropertyManager *) nHandle.get_rep());

	  nHandles.push_back( handle );
	}

	delete min;
	delete max;
       }
    } else if( pi != range.second ) {
      error( "No handle or representation" );
      return;
    }
  }

  string datasetsStr;

  // Get each of the dataset names for the GUI.
  for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {

    nHandle = nHandles[ic];

    // Get the tuple axis name - there is only one.
    vector< string > dataset;
    nHandle->get_tuple_indecies(dataset);

    // Save the name of the dataset.
    if( nHandles.size() == 1 )
      datasetsStr.append( dataset[0] );
    else
      datasetsStr.append( "{" + dataset[0] + "} " );
  }      


  if( datasetsStr != datasetsStr_.get() ) {
    // Update the dataset names and dims in the GUI.
    ostringstream str;
    str << id << " set_names " << " {" << datasetsStr << "}";
    
    gui->execute(str.str().c_str());
  }

  if( nHandles.size() != 1 && nHandles.size() != 3 &&
      nHandles.size() != 4 && nHandles.size() != 8 ){
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

    for( unsigned int ic=0; ic++; ic<nHandles.size() )
      nGenerations_[ic] = nHandles[ic]->generation;

    // Get each of the dataset names for the GUI.
    for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {

      nHandle = nHandles[ic];

      // Get the tuple axis name - there is only one.
      vector< string > dataset;
      nHandle->get_tuple_indecies(dataset);

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
		mesh_[R] = ic;
	      } else if( dataset[0].find( "Z:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 3 ) {
		conversion = MESH;
		mesh_[Z] = ic; 
	      } else if( dataset[0].find( "PHI:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 2 ) {
		mesh_[PHI] = ic;
	      } else if( dataset[0].find( "K:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 2 ) {
		conversion = PERTURBED;
		mesh_[K] = ic;
	      } else {
		error( dataset[0] + " is unknown NIMROD mesh data." );
		error_ = true;
		return;
	      }
	    } else {
	      error( dataset[0] + property + " is an unsupported coordinate system." );
	      error_ = true;
	      return;
	    }
	  } else {
	    error( dataset[0] + "No coordinate system found." );
	    error_ = true;
	    return;
	  }
	}
      } else if( nHandle->get_property( "DataSpace", property ) ) {

	if( property.find( "REALSPACE" ) != string::npos ) {

	  if( data_.size() == 0 ) {
	    data_.resize(3);
	    data_[0] = data_[1] = data_[2] = -1;
	  }

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
	  } else if( dataset[0].find( ":Scalar" ) != string::npos && 
		     nHandle->nrrd->dim == 4 ) {
	    conversion = SCALAR;
	    data_[0] = ic;
	  } else {
	    error( dataset[0] + " is unknown NIMROD node data." );
	    error_ = true;
	    return;
	  }
	} else if( property.find( "PERTURBED" ) != string::npos ) {

	  if( nHandle->get_property( "DataSubspace", property ) ) {
	    if( property.find( "REAL" ) != string::npos ||
		property.find( "IMAGINARY" ) != string::npos ) {

	      if( data_.size() == 0 ) {
		data_.resize(6);
		data_[0] = data_[1] = data_[2] = 
		  data_[3] = data_[4] = data_[5] = -1;
	      }

	      int offset = 0;
	      if( property.find( "REAL" ) != string::npos )
		offset = 0;
	      else if ( property.find( "IMAGINARY" ) != string::npos )
		offset = 1;
	      
	      int index = 0;
	      if( dataset[0].find( "R:Scalar" ) != string::npos &&
		  nHandle->nrrd->dim == 4 ) {
		conversion = PERTURBED;
		index = 0 + 3 * offset;
	      } else if( dataset[0].find( "Z:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 4 ) {
		conversion = PERTURBED;
		index = 1 + 3 * offset;
	      } else if( dataset[0].find( "PHI:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 4 ) {
		index = 2 + 3 * offset;
		conversion = PERTURBED;
	      } else { // Scalar data
		data_.resize(2);
		conversion = PERTURBED;
		index = 0 + 1 * offset;
	      }

	      data_[index] = ic;

	    } else {
	      error( dataset[0] + property + " Unsupported Data SubSpace." );
	      error_ = true;
	      return;
	    }
	  } else {
	    error( dataset[0] + " No Data SubSpace property." );
	    error_ = true;
	    return;
	  }
	} else {	
	  error( dataset[0] + property + " Unsupported Data Space." );
	  error_ = true;
	  return;
	}
      } else {
	error( dataset[0] + " No DataSpace property." );
	error_ = true;
	return;
      }
    }
  }

  unsigned int i = 0;
  while( i<data_.size() && data_[i] != -1 )
    i++;
  
  if( conversion & MESH ) {
    if( mesh_[0] == -1 || mesh_[1] == -1 || mesh_[2] == -1 ) {
      error( "Not enough mesh data for the mesh conversion." );
      error_ = true;
      return;
    }
  } else if ( conversion & REALSPACE ) {
    if( mesh_[2] == -1 || i != data_.size() ) {
      error( "Not enough data for the realspace conversion." );
      error_ = true;
      return;
    }
  } else if ( conversion & PERTURBED ) {

    if( mesh_[2] == -1 || mesh_[3] == -1 || i != data_.size() ) {
      error( "Not enough data for the perturbed conversion." );
      error_ = true;
      return;
    }
  }

  // If no data or data change, recreate the field.
  if( error_ ||
      !nHandle_.get_rep() ||
      mode_ != iMode_.get() ||
      generation ) {
    
    error_ = false;

    mode_ = iMode_.get();

    int idim=0, jdim=0, kdim=0;

    string convertStr;
    unsigned int ntype;

    if( conversion & MESH ) {
      ntype = nHandles[mesh_[PHI]]->nrrd->type;

      nHandles[mesh_[PHI]]->get_property( "Coordinate System", property );

      if( property.find("Cylindrical - NIMROD") != string::npos ) {
	if( nHandles[mesh_[R]]->nrrd->axis[1].size != 
	    nHandles[mesh_[Z]]->nrrd->axis[1].size ||
	    nHandles[mesh_[R]]->nrrd->axis[2].size != 
	    nHandles[mesh_[Z]]->nrrd->axis[2].size ) {
	  error( "Mesh dimension mismatch." );
	  error_ = true;
	  return;
	}

	idim = nHandles[mesh_[PHI]]->nrrd->axis[1].size; // Phi
	jdim = nHandles[mesh_[R]]->nrrd->axis[1].size; // Radial
	kdim = nHandles[mesh_[Z]]->nrrd->axis[2].size; // Theta
      }

      convertStr = "Mesh";

    } else if( conversion & SCALAR ) {
      ntype = nHandles[data_[0]]->nrrd->type;

      idim = nHandles[data_[0]]->nrrd->axis[1].size; // Phi
      jdim = nHandles[data_[0]]->nrrd->axis[2].size; // Radial
      kdim = nHandles[data_[0]]->nrrd->axis[3].size; // Theta

      convertStr = "Scalar";

     } else if( conversion & REALSPACE ) {
      ntype = nHandles[mesh_[PHI]]->nrrd->type;

      nHandles[mesh_[PHI]]->get_property( "Coordinate System", property );

      if( property.find("Cylindrical - NIMROD") != string::npos ) {
	idim = nHandles[mesh_[PHI]]->nrrd->axis[1].size; // Phi
	jdim = nHandles[data_[R]]->nrrd->axis[2].size; // Radial
	kdim = nHandles[data_[Z]]->nrrd->axis[3].size; // Theta

	for( unsigned int ic=0; ic<data_.size(); ic++ ) {
	  if( nHandles[data_[ic]]->nrrd->axis[1].size != idim ||
	      nHandles[data_[ic]]->nrrd->axis[2].size != jdim ||
	      nHandles[data_[ic]]->nrrd->axis[3].size != kdim ) {
	    error( "Mesh dimension mismatch." );
	    error_ = true;
	    return;
	  }
	}
      }

      convertStr = "RealSpace";

    } else if( conversion & PERTURBED ) {
      ntype = nHandles[mesh_[PHI]]->nrrd->type;

      nHandles[mesh_[PHI]]->get_property( "Coordinate System", property );

      int nmodes = 0;

      if( property.find("Cylindrical - NIMROD") != string::npos ) {
	nmodes = nHandles[mesh_[K]]->nrrd->axis[1].size; // Modes
	jdim   = nHandles[data_[0]]->nrrd->axis[2].size; // Radial
	kdim   = nHandles[data_[0]]->nrrd->axis[3].size; // Theta

	for( unsigned int ic=0; ic<data_.size(); ic++ ) {
	  if( nHandles[data_[ic]]->nrrd->axis[1].size != nmodes ||
	      nHandles[data_[ic]]->nrrd->axis[2].size != jdim ||
	      nHandles[data_[ic]]->nrrd->axis[3].size != kdim ) {
	    error( "Mesh dimension mismatch." );
	    error_ = true;
	    return;
	  }
	}
      }

      if( nmodes != nModes_.get() ) {
	// Update the dataset names and dims in the GUI.
	ostringstream str;
	str << id << " set_modes " << nmodes;
      
	gui->execute(str.str().c_str());

	remark( "Select the mode for the calculation" );
	error_ = true; // Not really an error but it so it will execute.
	return;
      } else {
	// This is cheat for passing but it gets updated anyways.
	idim = mode_;
      }

      convertStr = "Perturbed";
    }

    if( conversion ) {
      remark( "Converting the " + convertStr );
    
      CompileInfoHandle ci_mesh =
	NIMRODConverterAlgo::get_compile_info( convertStr, ntype );
    
      Handle<NIMRODConverterAlgo> algo_mesh;
    
      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;
      
      nHandle_ = algo_mesh->execute( nHandles, mesh_, data_, idim, jdim, kdim );
    } else {
      error( "Nothing to convert." );
      error_ = true;
      return;
    }
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

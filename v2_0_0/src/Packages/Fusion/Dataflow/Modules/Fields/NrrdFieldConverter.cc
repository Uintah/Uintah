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

#include <Fusion/Dataflow/Modules/Fields/NrrdFieldConverter.h>

#include <sci_defs.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class PSECORESHARE NrrdFieldConverter : public Module {
public:
  enum {UNKNOWN=0,UNSTRUCTURED=1,STRUCTURED=2,IRREGULAR=4,REGULAR=8};

public:
  NrrdFieldConverter(GuiContext*);

  virtual ~NrrdFieldConverter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

protected:
  GuiInt    noMesh_;
  GuiString datasetsStr_;

  vector< int > mesh_;
  vector< int > data_;

  unsigned int topology_;
  unsigned int geometry_;

  vector< int > nGenerations_;

  FieldHandle  fHandle_;

  bool error_;
};


DECLARE_MAKER(NrrdFieldConverter)
NrrdFieldConverter::NrrdFieldConverter(GuiContext* context)
  : Module("NrrdFieldConverter", context, Source, "Fields", "Fusion"),
    noMesh_(context->subVar("nomesh")),
    datasetsStr_(context->subVar("datasets")),

    topology_(UNKNOWN),
    geometry_(UNKNOWN),
    error_(false)
{
}

NrrdFieldConverter::~NrrdFieldConverter(){
}

void
NrrdFieldConverter::execute(){

  register unsigned int ic, jc, kc;

  reset_vars();

  MeshHandle mHandle;
  vector< NrrdDataHandle > nHandles;

  // Assume a range of ports even though only two are needed for the
  // mesh and data.
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

    NrrdDataHandle nHandle;

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
	for( ic=1; ic<nrrdDim; ic++) {
	  min[ic] = 0;
	  max[ic] = nHandle->nrrd->axis[ic].size-1;
	}

	// Separtate via the tupple axis.
	for( ic=0; ic<tuples; ic++ ) {

	  Nrrd *nout = nrrdNew();

	  // Translate the tuple index into the real offsets for a tuple axis.
	  int tmin, tmax;
	  if (! nHandle->get_tuple_index_info( ic, ic, tmin, tmax)) {
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
	  nout->axis[0].label = strdup(dataset[ic].c_str());

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

  if( nHandles.size() == 0 ){
    error( "No handle or representation" );
    return;
  }

  int generation = 0;

  // See if input data has been added or removed.
  if( nGenerations_.size() == 0 ||
      nGenerations_.size() != nHandles.size() )
    generation = nHandles.size();
  else {
    // See if any of the input data has changed.
    for( ic=0; ic<nHandles.size() && ic<nGenerations_.size(); ic++ ) {
      if( nGenerations_[ic] != nHandles[ic]->generation )
	++generation;
    }
  }

  string property;

  // If data change, update the GUI the field if needed and update the 
  // mesh and data lists.
  if( generation ) {

    // Save all of the current generation values.
    nGenerations_.resize( nHandles.size() );
    for( ic=0; ic++; ic<nHandles.size() )
      nGenerations_[ic] = nHandles[ic]->generation;

    topology_ = UNKNOWN;
    geometry_ = UNKNOWN;

    mesh_.clear();
    data_.clear();

    string datasetsStr;
    vector< string > datasets;

    // Get each of the dataset names for the GUI.
    for( ic=0; ic<nHandles.size(); ic++ ) {

      // Get the tuple axis name - there is only one.
      vector< string > dataset;
      nHandles[ic]->get_tuple_indecies(dataset);

      // Save the name of the dataset.
      if( nHandles.size() == 1 )
	datasetsStr.append( dataset[0] );
      else
	datasetsStr.append( "{" + dataset[0] + "} " );
      
      // Determined how each Nrrd is going to used.
      if( nHandles[ic]->get_property( "Topology", property ) ) {

	// Structured mesh.
	if( property.find( "Structured" ) != string::npos ) {

	  if( nHandles[ic]->get_property( "Geometry", property ) ) {
	    if( property.find( "Regular" ) != string::npos )
	      geometry_ = REGULAR;
	    else if( property.find( "Irregular" ) != string::npos )
	      geometry_ = IRREGULAR;
	  } else {
	    error( dataset[0] + " - Unknown geometry in mesh data found." );
	    error_ = true;
	    return;
	  }


	  if( nHandles[ic]->get_property( "Coordinate System", property ) ) {

	    // Cartesian Coordinates.
	    if( property.find("Cartesian") != string::npos ) {
	      mesh_.push_back( ic );	      
	      topology_ = STRUCTURED;

	    } else {
	      error( dataset[0] + " - " + property +
		     " is an unsupported coordinate system." );
	      error_ = true;
	      return;
	    }

	  } else {
	    error( dataset[0] + " - No coordinate system found." );
	    error_ = true;
	    return;
	  }

	  // Unstructured mesh.
	} else if( property.find( "Unstructured" ) != string::npos ) {

	  // For unstructured two lists are needed, points and cells.
	  if( mesh_.size() == 0 ) {
	    mesh_.resize( 2 );
	    mesh_[0] = mesh_[1] = -1;
	  }

	  // The cell list has two attributes:
	  // Topology == Unstructured and Cell Type == (see check below).
	  if( nHandles[ic]->get_property( "Cell Type", property ) ) {

	    if( mesh_[0] == -1 )
	      mesh_[0] = ic;
	    else {
	      error( dataset[0] +
		     " - Multiple connectivity lists not supported." );
	      error_ = true;
	      return;
	    }

	    topology_ = UNSTRUCTURED;

	    // The point list has two attributes:
	    // Topology == Unstructured and Coordinate System == Cartesian
	  } else if( nHandles[ic]->get_property( "Coordinate System", property ) ) {

	    // Cartesian Coordinates.
	    if( property.find("Cartesian") != string::npos ) {

	      if( mesh_[1] == -1 ) mesh_[1] = ic;
	      else                 mesh_.push_back( ic );

	      topology_ = UNSTRUCTURED;
	    } else {
	      error( dataset[0] + " - " + property +
		     " is an unsupported coordinate system." );
	      error_ = true;
	      return;
	    }
	  } else {
	    error( dataset[0] + " - Unknown unstructured mesh data found." );
	    error_ = true;
	    return;
	  }
	}
    } else
      // Anything else is considered to be data.
      data_.push_back( ic );
    }

    datasetsStr_.reset();
    
    if( datasetsStr != datasetsStr_.get() ) {
      // Update the dataset names and dims in the GUI.
      ostringstream str;
      str << id << " set_names {" << datasetsStr << "}";

      gui->execute(str.str().c_str());
    }
  }

  // See if any mesh data has been supplied.
  if( noMesh_.get() ) {
    if( data_.size() ) {
      if( mesh_.size() > 0 ) {
	warning( string( "Mesh data present but selected no mesh option. ") +
		 string("Ignoring mesh data.") );
	mesh_.clear();
      }
    
      topology_ = STRUCTURED;
      geometry_ = REGULAR;
    } else {
      error( "No data present to create a regular grid." );
      error_ = true;
      return;
    }
  } else if( mesh_.size() == 0 ) {
    error( "No mesh data present." );
    error_ = true;
    return;
  }

  // Check to make sure all of the needed mesh data has been supplied.
  for( ic=0; ic<mesh_.size(); ic++ ) {
    if( mesh_[ic] == -1 ) {
      if( topology_ & STRUCTURED ) {
	error( "Not enough information to create a mesh." );
	error_ = true;
	return;
      } else if( topology_ & UNSTRUCTURED && ic == 0 ) {
	warning( string( "No connection list for unstructured points,") +
		 string( "assuming a point cloud.") );
      }
    }
  }

  if( data_.size() && !topology_ ) {
    error( "Found data but no mesh organization." );
    error_ = true;
    return;
  }

  // If no data or data change, recreate the field.
  if( error_ ||
      !fHandle_.get_rep() ||
      generation ) {
 
    error_ = false;

    int idim=1, jdim=1, kdim=1;

    vector<unsigned int> mdims;
    int mesh_rank = 0;
    int mesh_coor_rank = 0;

    vector<unsigned int> ddims;
    int data_rank = 0;

    NrrdDataHandle dHandle;

    if( topology_ & STRUCTURED && geometry_ & REGULAR ) {

      Point minpt(0,0,0), maxpt(1,1,1);

      if( mesh_.size() == 0 ) {
	// No mesh data so assume 0->1 range.
      } else if( mesh_.size() == 1 ) {
	dHandle = nHandles[mesh_[0]];

	// Get the tuple axis name - there is only one.
	vector< string > dataset;
	dHandle->get_tuple_indecies(dataset);
	  
	if( dataset[0].find( ":Scalar" ) != string::npos ) {
	  mesh_rank = dHandle->nrrd->dim - 2;
	  mesh_coor_rank = dHandle->nrrd->axis[ dHandle->nrrd->dim-1].size;
	} else if( dataset[0].find( ":Vector" ) != string::npos ) {
	  mesh_rank = dHandle->nrrd->dim - 1;
	  mesh_coor_rank = 3;
	}
	  
	if( mesh_rank != 1 || mesh_coor_rank < 2 || 3 < mesh_coor_rank ) {
	  error( dataset[0] + " Mesh dataset does not contain points." );
	  error_ = true;
	  return;
	}
	  
	if( dHandle->nrrd->axis[ mesh_rank ].size != 2 ) {
	  error( dataset[0] + " Mesh dataset does not contain two points." );
	  error_ = true;
	  return;
	}
      } else if( mesh_.size() == 2 || mesh_.size() == 3 ) {

	mesh_coor_rank = mesh_.size();

	for( unsigned int ic=0; ic<mesh_.size(); ic++ ) {
	  dHandle = nHandles[mesh_[ic]];
	    
	  // Get the tuple axis name - there is only one.
	  vector< string > dataset;
	  dHandle->get_tuple_indecies(dataset);

	  if( dataset[0].find( ":Scalar" ) == string::npos ) {
	    error( dataset[0] + " - Data type must be scalar." );
	    error_ = true;
	    return;
	  }

	  if( ic == 0 ) {
	    mesh_rank = dHandle->nrrd->dim - 1;
	      
	    if( mesh_rank != 1 ) {
	      error( dataset[0] +
		     " Mesh dataset does not contain points." );
	      error_ = true;
	      return;
	    }
	  } else {
	    if( mesh_rank != dHandle->nrrd->dim-1 ) {
	      error( dataset[0] + " - Mesh rank mismatch." );
	      error_ = true;
	      return;
	    }
	  }

	  if( dHandle->nrrd->axis[ mesh_rank ].size != 2 ) {
	    error( dataset[0] +
		   " Mesh dataset does not contain two points." );
	    error_ = true;
	    return;
	  }
	}
      } else {
	error( "Mesh datasets do not contain points." );
	error_ = true;
	return;
      }


      if( data_.size() == 1 || data_.size() == 3 || data_.size() == 6 ) {

	for( ic=0; ic<data_.size(); ic++ ) {
	  dHandle = nHandles[data_[ic]];
	    
	  // Get the tuple axis name - there is only one.
	  vector< string > dataset;
	  dHandle->get_tuple_indecies(dataset);
	    
	  // If more than one dataset then all axii must be Scalar
	  if( data_.size() > 1 ) {
	    if( dataset[0].find( ":Scalar" ) == string::npos ) {
	      error( dataset[0] + " - Data type must be scalar." );
	      error_ = true;
	      return;
	    }
	  }


	  const unsigned int nrrdDim = dHandle->nrrd->dim;
	    
	  if( ic == 0 ) {
	    mesh_rank = nrrdDim - 1;
		
	    if( mesh_rank >= 1 ) idim = dHandle->nrrd->axis[1].size;
	    if( mesh_rank >= 2 ) jdim = dHandle->nrrd->axis[2].size;
	    if( mesh_rank >= 3 ) kdim = dHandle->nrrd->axis[3].size;

	    mdims.clear();
	    if( idim > 1) { mdims.push_back( idim ); }
	    if( jdim > 1) { mdims.push_back( jdim ); }
	    if( kdim > 1) { mdims.push_back( kdim ); }

	  } else {
	    if( mesh_rank != dHandle->nrrd->dim-1 ) {
	      error( dataset[0] + " - Mesh rank mismatch." );
	      error_ = true;
	      return;
	    }

	    for( jc=0, kc=0; jc<nrrdDim; jc++ )
	      if( dHandle->nrrd->axis[jc].size > 1 &&
		  mdims[kc++] != (unsigned int) dHandle->nrrd->axis[jc].size ) {
		error(  dataset[0] + "Data set sizes do not match." );
		    
		error_ = true;
		return;
	      }
	  }
	}
      } else {
	error( "Can not determine the mesh size from the datasets." );
	error_ = true;
	return;
      }

      // Create the mesh.
      if( mdims.size() == 3 ) {
	// 3D LatVolMesh
	mHandle =
	  scinew LatVolMesh( mdims[0], mdims[1], mdims[2], minpt, maxpt );
      } else if( mdims.size() == 2 ) {
	// 2D ImageMesh
	mHandle = scinew ImageMesh( mdims[0], mdims[1], minpt, maxpt );
      } else if( mdims.size() == 1 ) {
	// 1D ScanlineMesh
	mHandle = scinew ScanlineMesh( mdims[0], minpt, maxpt );
      } else {
	error( "Mesh dimensions do not make sense." );
	error_ = true;
	return;
      }

      if( mesh_.size() ) {
	const TypeDescription *mtd = mHandle->get_type_description();
    
	remark( "Creating a structured " + mtd->get_name() );

	CompileInfoHandle ci_mesh =
	  NrrdFieldConverterMeshAlgo::get_compile_info( "Regular",
					mtd,
					nHandles[mesh_[0]]->nrrd->type,
					nHandles[mesh_[0]]->nrrd->type);

	Handle<RegularNrrdFieldConverterMeshAlgo> algo_mesh;
    
	if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;
    
	algo_mesh->execute(mHandle, nHandles, mesh_);
      }
    } else if( topology_ & STRUCTURED && geometry_ & IRREGULAR ) {

      nHandles[mesh_[0]]->get_property( "Coordinate System", property );

      if( mesh_.size() == 1 ) {
	// Check to make sure there are two or three coordinates.
	// If Scalar the last dim must be two or three.
	// If already Vector then nothing ...
	  
	dHandle = nHandles[mesh_[0]];
	    
	// Get the tuple axis name - there is only one.
	vector< string > dataset;
	dHandle->get_tuple_indecies(dataset);

	if( dataset[0].find( ":Scalar" ) != string::npos ) {
	  mesh_rank = dHandle->nrrd->dim - 2;
	  mesh_coor_rank = dHandle->nrrd->axis[ dHandle->nrrd->dim-1].size;
	} else if( dataset[0].find( ":Vector" ) != string::npos ) {
	  mesh_rank = dHandle->nrrd->dim - 1;
	  mesh_coor_rank = 3;
	}
	  
	if( mesh_coor_rank < 2 || 3 < mesh_coor_rank ) {
	  error( dataset[0] + " Mesh dataset does not contain points." );
	  error_ = true;
	  return;
	}
	  
	if( mesh_rank >= 1 ) idim = nHandles[mesh_[0]]->nrrd->axis[1].size;
	if( mesh_rank >= 2 ) jdim = nHandles[mesh_[0]]->nrrd->axis[2].size;
	if( mesh_rank >= 3 ) kdim = nHandles[mesh_[0]]->nrrd->axis[3].size;
	  
	mdims.clear();		
	if( idim > 1) { mdims.push_back( idim ); }
	if( jdim > 1) { mdims.push_back( jdim ); }
	if( kdim > 1) { mdims.push_back( kdim ); }
	  
      } else if( mesh_.size() == 2 || mesh_.size() == 3 ) {

	mesh_coor_rank = mesh_.size();

	for( unsigned int ic=0; ic<mesh_.size(); ic++ ) {
	  dHandle = nHandles[mesh_[ic]];
	    
	  // Get the tuple axis name - there is only one.
	  vector< string > dataset;
	  dHandle->get_tuple_indecies(dataset);

	  if( dataset[0].find( ":Scalar" ) == string::npos ) {
	    error( dataset[0] + " - Data type must be scalar." );
	    error_ = true;
	    return;
	  }

	  if( ic == 0 ) {
	    mesh_rank = dHandle->nrrd->dim - 1;
	      
	    if( mesh_rank >= 1 ) idim = dHandle->nrrd->axis[1].size;
	    if( mesh_rank >= 2 ) jdim = dHandle->nrrd->axis[2].size;
	    if( mesh_rank >= 3 ) kdim = dHandle->nrrd->axis[3].size;

	    mdims.clear();
	    if( idim > 1) { mdims.push_back( idim ); }
	    if( jdim > 1) { mdims.push_back( jdim ); }
	    if( kdim > 1) { mdims.push_back( kdim ); }

	  } else {
	    if( mesh_rank != dHandle->nrrd->dim-1 ) {
	      error( dataset[0] + " - Mesh rank mismatch." );
	      error_ = true;
	      return;
	    }

	    for( int jc=0, kc=0; jc<dHandle->nrrd->dim; jc++ )
	      if( dHandle->nrrd->axis[jc].size > 1 &&
		  mdims[kc++] != (unsigned int) dHandle->nrrd->axis[jc].size ) {
		error(  dataset[0] + "Data set sizes do not match." );
		    
		error_ = true;
		return;
	      }
	  }
	}
      } else {
	error( "Mesh datasets do not contain points." );
	error_ = true;
	return;
      }

      // Create the mesh.
      if( mdims.size() == 3 ) {
	// 3D StructHexVol
	mHandle = scinew StructHexVolMesh( mdims[0], mdims[1], mdims[2] );
      } else if( mdims.size() == 2 ) {
	// 2D StructQuadSurf
	mHandle = scinew StructQuadSurfMesh( mdims[0], mdims[1] );
      } else if( mdims.size() == 1 ) {
	// 1D StructCurveMesh
	mHandle = scinew StructCurveMesh( mdims[0] );
      } else {
	error( "Mesh dimensions do not make sense." );
	error_ = true;
	return;
      }

      const TypeDescription *mtd = mHandle->get_type_description();
    
      remark( "Creating a structured " + mtd->get_name() );

      CompileInfoHandle ci_mesh =
	NrrdFieldConverterMeshAlgo::get_compile_info( "Structured",
				      mtd,
				      nHandles[mesh_[0]]->nrrd->type,
				      nHandles[mesh_[0]]->nrrd->type);

      Handle<StructuredNrrdFieldConverterMeshAlgo> algo_mesh;
    
      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;
    
      algo_mesh->execute(mHandle, nHandles, mesh_,
			 idim, jdim, kdim);
    
    } else if( topology_ & UNSTRUCTURED ) {

      NrrdDataHandle cHandle = NULL;
      NrrdDataHandle pHandle = NULL;

      if( mesh_.size() == 2 ) {
	// Check to make sure there are two or three coordinates.
	// If Scalar the last dim must be two or three.
	// If already Vector then nothing ...
	
	pHandle = nHandles[mesh_[1]];
	
	// Get the tuple axis name - there is only one.
	vector< string > dataset;
	pHandle->get_tuple_indecies(dataset);

	if( pHandle->nrrd->dim == 3 &&
	    dataset[0].find( ":Scalar" ) != string::npos )
	  mesh_coor_rank = pHandle->nrrd->axis[2].size;
	else if( pHandle->nrrd->dim == 2 &&
		 dataset[0].find( ":Vector" ) != string::npos )
	  mesh_coor_rank = 3;
	      
	if( mesh_coor_rank < 2 || 3 < mesh_coor_rank ) {
	  error( dataset[0] + " - Mesh does not contain points." );
	  error_ = true;
	  return;
	}

	mdims.clear();
	mdims.push_back( pHandle->nrrd->axis[1].size );

      } else if( mesh_.size() == 3 || mesh_.size() == 4 ) {

	mesh_coor_rank = mesh_.size() - 1;

	for( unsigned int ic=1; ic<mesh_.size(); ic++ ) {
	  dHandle = nHandles[mesh_[ic]];
	    
	  // Get the tuple axis name - there is only one.
	  vector< string > dataset;
	  dHandle->get_tuple_indecies(dataset);

	  if( dHandle->nrrd->dim == 2 &&
	      dataset[0].find( ":Scalar" ) == string::npos ) {
	    error( dataset[0] + " - Data type must be scalar." );
	    error_ = true;
	    return;
	  }

	  if( ic == 1 ) {
	    pHandle = dHandle;
	  } else if( pHandle->nrrd->axis[1].size != 
		     dHandle->nrrd->axis[1].size ) {
	    error( dataset[0] + " - Mesh size mismatch." );
	    error_ = true;
	    return;
	  }
	}
	
	mdims.clear();
	mdims.push_back( pHandle->nrrd->axis[1].size );

      } else {
	error( "Mesh datasets do not contain points." );
	error_ = true;
	return;
      }

      int connectivity = 0;

      if( mesh_[0] >= 0 ) {

	cHandle = nHandles[mesh_[0]];
	
	// Get the tuple axis name - there is only one.
	vector< string > dataset;
	pHandle->get_tuple_indecies(dataset);

	if( !(cHandle->nrrd->dim == 2 &&
	      (dataset[0].find( ":Vector" ) != string::npos ||
	       dataset[0].find( ":Tensor" ) != string::npos)) &&
	    
	    !(cHandle->nrrd->dim == 3 &&
	      dataset[0].find( ":Scalar" ) != string::npos) ) {
	  error( dataset[0] + " - Malformed connectivity list." );
	  error_ = true;
	  return;
	}

	if( cHandle->get_property( "Cell Type", property ) ) {
	  if( property.find( "Tet" ) != string::npos ) {
	    mHandle = scinew TetVolMesh();
	    connectivity = 4;
	  } else if( 0 && property.find( "Pyramid" ) != string::npos ) {
	    //mHandle = scinew PyramidVolMesh();
	    connectivity = 5;
	  } else if( property.find( "Prism" ) != string::npos ) {
	    mHandle = scinew PrismVolMesh();
	    connectivity = 6;
	  } else if( property.find( "Hex" ) != string::npos ) {
	    mHandle = scinew HexVolMesh();
	    connectivity = 8;
	  } else if( property.find( "Tri" ) != string::npos ) {
	    mHandle = scinew TriSurfMesh();
	    connectivity = 3;
	  } else if( property.find( "Quad" ) != string::npos ) {
	    mHandle = scinew QuadSurfMesh();
	    connectivity = 4;
	  } else if( property.find( "Curve" ) != string::npos ) {
	    mHandle = scinew CurveMesh();
	    connectivity = 2;
	  } else if( property.find( "Point" ) != string::npos ) {
	    mHandle = scinew PointCloudMesh();
	    connectivity = 0;
	  }
	  else {
	    error( dataset[0] + property + " Unsupported cell type." );
	    error_ = true;
	    return;
	  }

	  // If the dim is three then the last axis has the number
	  // of connections.
	  if( !(cHandle->nrrd->dim == 3 &&
		cHandle->nrrd->axis[2].size == connectivity) &&
	      
	      // If stored as Vector or Tensor the last dim will be the point
	      // list dim so do this check instead.
	      !(connectivity == 3 &&
		dataset[0].find( ":Vector" ) != string::npos ) &&
	      !(connectivity == 6 &&
		dataset[0].find( ":Tensor" ) != string::npos ) ) {

	    error( dataset[0] + "- Connectivity list set does not contain enough points for the cell type." );
	    error_ = true;
	    return;
	  }
	}
      }
      else {
	mHandle = scinew PointCloudMesh();
	connectivity = 0;
      }

      const TypeDescription *mtd = mHandle->get_type_description();
      
      remark( "Creating an unstructured " + mtd->get_name() );

      CompileInfoHandle ci_mesh =
	NrrdFieldConverterMeshAlgo::get_compile_info("Unstructured",
						mtd,
						pHandle->nrrd->type,
						connectivity ?
						cHandle->nrrd->type : 0 );
      
      Handle<UnstructuredNrrdFieldConverterMeshAlgo> algo_mesh;
      
      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;
      
      algo_mesh->execute(mHandle, nHandles, mesh_, connectivity);
    }

    
    // Set the rank of the data Scalar(1), Vector(3), Tensor(6).
    // Assume all of the input nrrd data is scalar with the last axis
    // size being the rank of the data.

    if( data_.size() == 0 ) {
      dHandle = NULL;
      data_rank = 1;
    } else if( data_.size() == 1 || data_.size() == 3 || data_.size() == 6 ) {

      for( unsigned int ic=0; ic<data_.size(); ic++ ) {
	dHandle = nHandles[data_[ic]];
	
	// Get the tuple axis name - there is only one.
	vector< string > dataset;
	dHandle->get_tuple_indecies(dataset);

	// If more than one dataset then all axii must be Scalar
	if( data_.size() > 1 ) {
	  if( dataset[0].find( ":Scalar" ) == string::npos ) {
	    error( dataset[0] + " - Data type must be scalar." );
	    error_ = true;
	    return;
	  }
	}

	if( ic == 0 ) {
	  ddims.clear();

	  for( int jc=1; jc<dHandle->nrrd->dim; jc++ )
	    if( dHandle->nrrd->axis[jc].size > 1 )
	      ddims.push_back( dHandle->nrrd->axis[jc].size );
	} else {

	  for( int jc=1, kc=0; jc<dHandle->nrrd->dim; jc++ )
	    if( dHandle->nrrd->axis[jc].size > 1 &&
		ddims[kc++] != (unsigned int) dHandle->nrrd->axis[jc].size ) {
	      error(  dataset[0] + "Data set sizes do not match." );

	      error_ = true;
	      return;
	    }
	}

	if( ddims.size() == mdims.size() ||
	    ddims.size() == mdims.size() + 1 ) {

	  for( unsigned int jc=0; jc<mdims.size(); jc++ ) {
	    if( ddims[jc] != mdims[jc] ) {
	      error(  dataset[0] + "Mesh and Data sizes do not match." );

	      {
		ostringstream str;

		str << " Mesh dimensions: ";
		for( unsigned int jc=0; jc<mdims.size(); jc++ )
		  str << mdims[jc] << "  ";
		error( str.str() );
	      }

	      {
		ostringstream str;

		for( unsigned int jc=0; jc<ddims.size(); jc++ )
		  str << ddims[jc] << "  ";
		error( str.str() );
	      }

	      error_ = true;
	      return;
	    }
	  }
	} else {
	  error( dataset[0] + "Mesh and Data are not of the same rank." );

	  ostringstream str;

	  str << "Mesh rank " << mdims.size() << "  ";
	  str << "Data rank " << ddims.size();
	  error( str.str() );
	  error_ = true;
	  return;
	}
      }

      if( data_.size() == 1 ) { 

	// Get the tuple axis name - there is only one.
	vector< string > dataset;
	nHandles[data_[0]]->get_tuple_indecies(dataset);

	if( ddims.size() == mdims.size() ) {
	  if( dataset[0].find( ":Scalar" ) != string::npos )
	    data_rank = 1;
	  else if( dataset[0].find( ":Vector" ) != string::npos )
	    data_rank = 3;
	  else if( dataset[0].find( ":Tensor" ) != string::npos )
	    data_rank = 6;
 	  else {
	    error( dataset[0] + "Bad tuple axis - no data type must be scalar, vector, or tensor." );
	    error_ = true;
	    return;
	  }
	} else if(ddims.size() == mdims.size() + 1) {
	  data_rank = ddims[mdims.size()];
	}
      } else {
	data_rank = data_.size();
      }
    } else {
      error( "Impropper number of data handles." );
      error_ = true;
      return;

    }

    if( data_rank != 1 && data_rank != 3 && data_rank != 6 ) {
      error( "Bad data rank." );
      error_ = true;
      return;
    }

    if( data_.size() )
      remark( "Adding in the data." );

    const TypeDescription *mtd = mHandle->get_type_description();
    
    string fname = mtd->get_name();
    string::size_type pos = fname.find( "Mesh" );
    fname.replace( pos, 4, "Field" );
    
    CompileInfoHandle ci =
      NrrdFieldConverterFieldAlgo::get_compile_info(mtd,
					       fname,
					       data_.size() ? 
					       nHandles[data_[0]]->nrrd->type : 0,
					       data_rank);
    
    Handle<NrrdFieldConverterFieldAlgo> algo;
    
    if (!module_dynamic_compile(ci, algo)) return;
    
    if( topology_ & STRUCTURED ) {
      fHandle_ = algo->execute( mHandle, nHandles, data_,
			        idim, jdim, kdim );

    } else if( topology_ & UNSTRUCTURED ) {
      fHandle_ = algo->execute( mHandle, nHandles, data_ );
    }
  }

  // Get a handle to the output field port.
  if( fHandle_.get_rep() ) {
    FieldOPort *ofield_port = 
      (FieldOPort *) get_oport("Output Field");
    
    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( fHandle_ );

  }
}

void
NrrdFieldConverter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


void get_nrrd_compile_type( const unsigned int type,
			    string & typeStr,
			    string & typeName )
{
  switch (type) {
  case nrrdTypeChar :  
    typeStr = string("char");
    typeName = string("char");
    break;
  case nrrdTypeUChar : 
    typeStr = string("unsigned char");
    typeName = string("unsigned_char");
    break;
  case nrrdTypeShort : 
    typeStr = string("short");
    typeName = string("short");
    break;
  case nrrdTypeUShort :
    typeStr = string("unsigned short");
    typeName = string("unsigned_short");
    break;
  case nrrdTypeInt : 
    typeStr = string("int");
    typeName = string("int");
    break;
  case nrrdTypeUInt :  
    typeStr = string("unsigned int");
    typeName = string("unsigned_int");
    break;
  case nrrdTypeLLong : 
    typeStr = string("long long");
    typeName = string("long_long");
    break;
  case nrrdTypeULLong :
    typeStr = string("unsigned long long");
    typeName = string("unsigned_long_long");
    break;
  case nrrdTypeFloat :
    typeStr = string("float");
    typeName = string("float");
    break;
  case nrrdTypeDouble :
    typeStr = string("double");
    typeName = string("double");
    break;
  default:
    typeStr = string("float");
    typeName = string("float");
  }
}

CompileInfoHandle
NrrdFieldConverterMeshAlgo::get_compile_info( const string topoStr,
					      const TypeDescription *mtd,
					      const unsigned int ptype,
					      const unsigned int ctype)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string base_class_name(topoStr + "NrrdFieldConverterMeshAlgo");
  const string template_class_name(topoStr + "NrrdFieldConverterMeshAlgoT");

  string pTypeStr,  cTypeStr;
  string pTypeName, cTypeName;

  get_nrrd_compile_type( ptype, pTypeStr, pTypeName );
  get_nrrd_compile_type( ctype, cTypeStr, cTypeName );

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mtd->get_filename() + "." +
		       pTypeName + "." + cTypeName + ".",
                       base_class_name, 
                       template_class_name,
                       mtd->get_name() + ", " + pTypeStr + ", " + cTypeStr );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  mtd->fill_compile_info(rval);
  return rval;
}

CompileInfoHandle
NrrdFieldConverterFieldAlgo::get_compile_info(const TypeDescription *mtd,
					      const string fname,
					      const unsigned int type,
					      int rank)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("NrrdFieldConverterFieldAlgo");

  string typeStr, typeName;

  get_nrrd_compile_type( type, typeStr, typeName );

  string extension;
  switch (rank)
  {
  case 6:
    extension = "Tensor";
    break;

  case 3:
    extension = "Vector";
    break;

  default:
    extension = "Scalar";
    break;
  }

  CompileInfo *rval = 
    scinew CompileInfo(base_class_name + extension + "." +
		       mtd->get_filename() + "." + 
		       typeName + ".",
                       base_class_name,
                       base_class_name + extension, 
		       fname +
		       "< " + (rank==1 ? typeName : extension) + " >" + ", " + 
		       mtd->get_name() + ", " + 
		       typeStr );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  mtd->fill_compile_info(rval);
  return rval;
}

} // End namespace Fusion

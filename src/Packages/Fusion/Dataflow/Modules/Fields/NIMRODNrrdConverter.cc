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
 *  NIMRODNrrdConverter.cc:
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

#include <Packages/Fusion/Dataflow/Modules/Fields/NIMRODNrrdConverter.h>

#include <sci_defs.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class PSECORESHARE NIMRODNrrdConverter : public Module {
public:
  NIMRODNrrdConverter(GuiContext*);

  virtual ~NIMRODNrrdConverter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString datasetsStr_;

  GuiInt iWrap_;
  GuiInt jWrap_;
  GuiInt kWrap_;

  int iwrap_;
  int jwrap_;
  int kwrap_;

  vector< int > grid_;
  vector< int > data_;

  vector< int > nGenerations_;

  FieldHandle  fHandle_;

  bool error_;
};


DECLARE_MAKER(NIMRODNrrdConverter)
NIMRODNrrdConverter::NIMRODNrrdConverter(GuiContext* context)
  : Module("NIMRODNrrdConverter", context, Source, "Fields", "Fusion"),
    datasetsStr_(context->subVar("datasets")),

    iWrap_(context->subVar("i-wrap")),
    jWrap_(context->subVar("j-wrap")),
    kWrap_(context->subVar("k-wrap")),

    iwrap_(0),
    jwrap_(0),
    kwrap_(0),

    error_(false)
{
}

NIMRODNrrdConverter::~NIMRODNrrdConverter(){
}

void
NIMRODNrrdConverter::execute(){

  vector< NrrdDataHandle > nHandles;
  NrrdDataHandle nHandle;
  MeshHandle mHandle;

  // Assume a range of ports even though only two are needed for the
  // grid and data.
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

  if( nHandles.size() == 0 ){
    error( "No handle or representation" );
    return;
  }

  if( nHandles.size() != 3 &&
      nHandles.size() != 4 &&
      nHandles.size() != 6 &&
      nHandles.size() != 9 ) {
    remark( "Too many inputs, ignoring extras." );
    return;
  }

  int generation = 0;

  // See if input data has been added or removed.
  if( nGenerations_.size() == 0 ||
      nGenerations_.size() != nHandles.size() )
    generation = nHandles.size();
  else {
    // See if any of the input data has changed.
    for( unsigned int ic=0; ic<nHandles.size() && ic<nGenerations_.size(); ic++ ) {
      if( nGenerations_[ic] != nHandles[ic]->generation )
	++generation;
    }
  }

  // If data change, update the GUI the field if needed.
  if( generation ) {

    nGenerations_.resize( nHandles.size() );
    grid_.resize( 3 );
    data_.clear();

    grid_[0] = grid_[1] = grid_[2] = -1;

    for( unsigned int ic=0; ic++; ic<nHandles.size() ) {
      nGenerations_[ic] = nHandles[ic]->generation;
    }

    string datasetsStr;

    vector< string > datasets;

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

      if( dataset[0].find( "GRID-R" ) != std::string::npos &&
	  nHandle->nrrd->dim == 3 ) 
	grid_[0] = ic;
      else if( dataset[0].find( "GRID-Z" ) != std::string::npos && 
	       nHandle->nrrd->dim == 3 ) 
	grid_[1] = ic; 
      else if( dataset[0].find( "GRID-PHI" ) != std::string::npos &&
	       nHandle->nrrd->dim == 2 )
	grid_[2] = ic;
      else
	data_.push_back( ic );
    }

    if( datasetsStr != datasetsStr_.get() ) {
      // Update the dataset names and dims in the GUI.
      ostringstream str;
      str << id << " set_names { " << datasetsStr << "}";
      
      gui->execute(str.str().c_str());
    }
  }


  if( grid_.size() != 3 ||
      grid_[0] == -1 || grid_[1] == -1 || grid_[2] == -1 ) {
    error( "Can not form the grid, missing components." );
    error_ = true;
    return;
  }

  // If no data or data change, recreate the field.
  if( error_ ||
      !fHandle_.get_rep() ||
      generation ||
      iwrap_ != iWrap_.get() ||
      jwrap_ != jWrap_.get() ||
      kwrap_ != kWrap_.get() ) {
 
    error_ = false;

    iwrap_ = iWrap_.get();
    jwrap_ = jWrap_.get();
    kwrap_ = kWrap_.get();


    if( nHandles[grid_[0]]->nrrd->axis[1].size != 
	nHandles[grid_[1]]->nrrd->axis[1].size ||
	nHandles[grid_[0]]->nrrd->axis[2].size != 
	nHandles[grid_[1]]->nrrd->axis[2].size ) {
      error( "Grid dimension mismatch." );
      error_ = true;
      return;
    }

    int idim = nHandles[grid_[2]]->nrrd->axis[1].size;
    int jdim = nHandles[grid_[0]]->nrrd->axis[1].size;
    int kdim = nHandles[grid_[0]]->nrrd->axis[2].size;

    vector<unsigned int> mdims;

    remark( "Creating the mesh." );

    // Create the mesh.
    if( idim > 1 && jdim > 1 && kdim > 1 ) {
      // 3D StructHexVol
      mHandle = scinew
	StructHexVolMesh( kdim+kwrap_, jdim+jwrap_, idim+iwrap_ );

      mdims.push_back( idim );
      mdims.push_back( kdim );
      mdims.push_back( jdim );

    } else if( idim >  1 && jdim >  1 && kdim == 1 ) {
      // 2D StructQuadSurf
      mHandle = scinew StructQuadSurfMesh(jdim+jwrap_, idim+iwrap_ );

      mdims.push_back( jdim );
      mdims.push_back( idim );
      kwrap_ = 0;

    } else if( idim >  1 && jdim == 1 && kdim  > 1 ) {
      // 2D StructQuadSurf
      mHandle = scinew StructQuadSurfMesh(kdim+kwrap_, idim+iwrap_ );

      mdims.push_back( kdim );
      mdims.push_back( idim );
      jwrap_ = 0;

    } else if( idim == 1 && jdim >  1 && kdim  > 1 ) {
      // 2D StructQuadSurf
      mHandle = scinew StructQuadSurfMesh(kdim+kwrap_, jdim+jwrap_ );

      mdims.push_back( kdim );
      mdims.push_back( jdim );
      iwrap_ = 0;

    } else if( idim  > 1 && jdim == 1 && kdim == 1 ) {
      // 1D StructCurveMesh
      mHandle = scinew StructCurveMesh( idim+iwrap_ );

      mdims.push_back( idim );
      jwrap_ = kwrap_ = 0;

    } else if( idim == 1 && jdim  > 1 && kdim == 1 ) {
      // 1D StructCurveMesh
      mHandle = scinew StructCurveMesh( jdim+jwrap_ );

      mdims.push_back( jdim );
      iwrap_ = kwrap_ = 0;

    } else if( idim == 1 && jdim == 1 && kdim  > 1 ) {
      // 1D StructCurveMesh
      mHandle = scinew StructCurveMesh( kdim+kwrap_ );

      mdims.push_back( kdim );
      iwrap_ = jwrap_ = 0;

    } else {
      error( "Grid dimensions do not make sense." );
      error_ = true;
      return;
    }
    
    const TypeDescription *mtd = mHandle->get_type_description();
      
    CompileInfoHandle ci_mesh =
      NIMRODNrrdConverterMeshAlgo::get_compile_info(mtd, nHandle->nrrd->type);

    Handle<NIMRODNrrdConverterMeshAlgo> algo_mesh;
      
    if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;

    algo_mesh->execute(mHandle, nHandles, grid_,
		       idim, jdim, kdim,
		       iwrap_, jwrap_, kwrap_);

    int ndims = mHandle.get_rep()->dimensionality();

    
    // Set the rank of the data Scalar(1), Vector(3), Tensor(6).
    // Assume all of the input nrrd data is scalar with the last axis
    // size being the rank of the data.

    vector<unsigned int> ddims;
    int rank = 0;
    if( data_.size() == 0 ) {
      nHandle = NULL;
      rank = 1;
    } else if( data_.size() == 1 || data_.size() == 3 || data_.size() == 6 ) {

      for( unsigned int ic=0; ic<data_.size(); ic++ ) {
	nHandle = nHandles[data_[ic]];
	
	int tuples = nHandle->get_tuple_axis_size();
	
	if( tuples != 1 ) {
	  error( "Too many tuples listed in the tuple axis." );
	  error_ = true;
	  return;
	}

	// Do not allow joined Nrrds
	vector< string > dataset;

	nHandle->get_tuple_indecies(dataset);

	if( dataset.size() != 1 ) {
	  error( "Too many sets listed in the tuple axis." );
	  error_ = true;
	  return;
	}

	if( data_.size() == 3 || data_.size() == 6 ) {
	  if( dataset[0].find( ":Scalar" ) == std::string::npos ) {
	    error( "Bad tuple axis - data type must be scalar. Found:" );
	    error( dataset[0] );
	    error_ = true;
	    return;
	  }
	}
	    
	for( int jc=nHandle->nrrd->dim-1; jc>0; jc-- )
	  if( nHandle->nrrd->axis[jc].size != 1 )
	    ddims.push_back( nHandle->nrrd->axis[jc].size );

	if( ddims.size() == mdims.size() ||
	    ddims.size() == mdims.size() + 1 ) {

	  for( unsigned int jc=0; jc<mdims.size(); jc++ ) {
	    if( ddims[jc] != mdims[jc] ) {
	      error( "Data and grid sizes do not match." );
	      error_ = true;
	      return;
	    }
	  }
	} 
      }

      if( data_.size() == 1 ) { 
	vector< string > dataset;

	nHandle->get_tuple_indecies(dataset);

	if( ddims.size() == mdims.size() ) {
	  if( dataset[0].find( ":Scalar" ) != std::string::npos )
	    rank = 1;
	  else if( dataset[0].find( ":Vector" ) != std::string::npos )
	    rank = 3;
	  else if( dataset[0].find( ":Tensor" ) != std::string::npos )
	    rank = 6;
 	  else {
	    error( "Bad tuple axis - no data type must be scalar, vector, or tensor." );
	    error( dataset[0] );
	    error_ = true;
	    return;
	  }
	}
	else if(ddims.size() == mdims.size() + 1) {

	  if( dataset[0].find( ":Scalar" ) != std::string::npos )
	    rank = ddims[mdims.size()];
	  else {
	    error( "Bad tuple axis - data type must be scalar. Found:" );
	    error( dataset[0] );
	    error_ = true;
	    return;
	  }
	} else {
	  error( "Data and grid dimensions do not match." );
	  error( dataset[0] );
	  error_ = true;
	}
      } else {
	rank = data_.size();
      }
    } else {
      error( "Impropper number of data handles." );
      error_ = true;
      return;

    }

    if( rank != 1 && rank != 3 && rank != 6 ) {
      error( "Bad data rank." );
      error_ = true;
      return;
    }

    remark( "Adding in the data." );

    // Add the data.
    if( ndims == 3 ) {
      // 3D StructHexVol
      StructHexVolMesh *mesh = (StructHexVolMesh *) mHandle.get_rep();
	
      if( rank == 1 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructHexVolField<float>(mesh, Field::NODE);
      } else if( rank == 3 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructHexVolField< Vector >(mesh, Field::NODE);
      } else if( rank == 6 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructHexVolField< Tensor >(mesh, Field::NODE);
      }
    } else if( ndims == 2 ) {
      // 2D StructQuadSurf
      StructQuadSurfMesh *mesh = (StructQuadSurfMesh *) mHandle.get_rep();

      if( rank == 1 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructQuadSurfField<float >(mesh, Field::NODE);
      } else if( rank == 3 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructQuadSurfField< Vector >(mesh, Field::NODE);
      } else if( rank == 6 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructQuadSurfField< Tensor >(mesh, Field::NODE);
      }
    } else if( ndims == 1 ) {
      // 1D StructCurve
      StructCurveMesh *mesh = (StructCurveMesh *) mHandle.get_rep();

      if( rank == 1 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructCurveField<float >(mesh, Field::NODE);
      } else if( rank == 3 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructCurveField< Vector >(mesh, Field::NODE);
      } else if( rank == 6 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructCurveField< Tensor >(mesh, Field::NODE);
      }
    } else {
      error( "Data dimensions do not make sense." );
      error_ = true;
      return;
    }

    if( data_.size() ) {

      // For vectors phi of the grid is needed.
      data_.push_back( grid_[2] );

      const TypeDescription *ftd = fHandle_->get_type_description();
      
      CompileInfoHandle ci =
	NIMRODNrrdConverterFieldAlgo::get_compile_info(ftd, nHandle->nrrd->type, rank);
      
      Handle<NIMRODNrrdConverterFieldAlgo> algo;

      if (!module_dynamic_compile(ci, algo)) return;

      algo->execute(fHandle_, nHandles, data_,
		    idim, jdim, kdim,
		    iwrap_, jwrap_, kwrap_ );
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
 NIMRODNrrdConverter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}



CompileInfoHandle
NIMRODNrrdConverterMeshAlgo::get_compile_info(const TypeDescription *ftd,
			       const unsigned int type)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("NIMRODNrrdConverterMeshAlgo");
  static const string template_class_name("NIMRODNrrdConverterMeshAlgoT");

  string typeStr;
  string typeName;

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
  }

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." + typeName + ".",
                       base_class_name, 
                       template_class_name,
                       ftd->get_name() + ", " + typeStr );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
 rval->add_namespace("Fusion");
  ftd->fill_compile_info(rval);
  return rval;
}

CompileInfoHandle
NIMRODNrrdConverterFieldAlgo::get_compile_info(const TypeDescription *ftd,
				const unsigned int type,
				int rank)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("NIMRODNrrdConverterFieldAlgo");

  string typeStr;
  string typeName;

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
  }

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
		       ftd->get_filename() + "." + typeName + ".",
                       base_class_name,
                       base_class_name + extension, 
                       ftd->get_name() + ", " + typeStr );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace Fusion



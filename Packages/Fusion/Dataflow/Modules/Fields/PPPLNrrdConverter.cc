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
 *  PPPLNrrdConverter.cc:
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

#include <Packages/Fusion/Dataflow/Modules/Fields/PPPLNrrdConverter.h>

#include <sci_defs.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class PSECORESHARE PPPLNrrdConverter : public Module {
public:
  PPPLNrrdConverter(GuiContext*);

  virtual ~PPPLNrrdConverter();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString datasetsStr_;
  GuiString datadimsStr_;

  GuiInt gridData_;

  GuiInt iWrap_;
  GuiInt jWrap_;
  GuiInt kWrap_;

  int grid_;
  int data_;

  int iwrap_;
  int jwrap_;
  int kwrap_;

  bool error_;

  vector< int > nGenerations_;

  FieldHandle  fHandle_;
};


DECLARE_MAKER(PPPLNrrdConverter)
PPPLNrrdConverter::PPPLNrrdConverter(GuiContext* context)
  : Module("PPPLNrrdConverter", context, Source, "Fields", "Fusion"),
    datasetsStr_(context->subVar("datasets")),
    datadimsStr_(context->subVar("datadims")),

    gridData_(context->subVar("grid")),

    iWrap_(context->subVar("i-wrap")),
    jWrap_(context->subVar("j-wrap")),
    kWrap_(context->subVar("k-wrap")),

    grid_(-1),
    data_(-1),

    iwrap_(0),
    jwrap_(0),
    kwrap_(0),

    error_( false )
{
}

PPPLNrrdConverter::~PPPLNrrdConverter(){
}

void
PPPLNrrdConverter::execute(){

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

  if( nHandles.size() > 2 ) {
    remark( "Too many inputs, ignoring extras." );
  }

  int generation = 0;

  // See if input data has been added or removed.
  if( nGenerations_.size() == 0 ||
      nGenerations_.size() != nHandles.size() )
    generation = nHandles.size();
  else {
    // See if any of the input data has changed.
    for( int ic=0; ic<nHandles.size() && ic<nGenerations_.size(); ic++ ) {
      if( nGenerations_[ic] != nHandles[ic]->generation )
	++generation;
    }
  }

  // If data change, update the GUI the field if needed.
  if( generation ) {

    nGenerations_.resize( nHandles.size() );

    for( int ic=0; ic++; ic<nHandles.size() ) {
      nGenerations_[ic] = nHandles[ic]->generation;
    }

    string datasetsStr;
    string datadimsStr;

    // Get each of the dataset names for the GUI.
    for( int ic=0; ic<nHandles.size(); ic++ ) {

      nHandle = nHandles[ic];

      vector< string > datasets;

      int tuples = nHandle->get_tuple_axis_size();
      nHandle->get_tuple_indecies(datasets);

      // Do not allow joined Nrrds
      if( datasets.size() != 1 ) {
	error( "Too many sets listed in the tuple axis." );
	error_ = true;
	return;
      }

      // Save the name of the dataset.
      if( nHandles.size() == 1 )
	datasetsStr.append( datasets[0] );
      else
	datasetsStr.append( "{" + datasets[0] + "} " );
      
      // Save the dimension of the dataset.
      char dimStr[8];
      sprintf( dimStr, "%d ", nHandle->nrrd->dim );
      
      datadimsStr.append( dimStr );
    }
  
    if( datasetsStr != datasetsStr_.get() ||
	datadimsStr != datadimsStr_.get() ) {
      // Update the dataset names and dims in the GUI.
      ostringstream str;
      str << id << " set_names { " << datasetsStr << "}";
      str << " {" << datadimsStr << "}";
      
      gui->execute(str.str().c_str());
    }
  }

  // Make sure a dataset for the grid has been selected.
  if( gridData_.get() < 0 ) {
    warning( "No grid dataset selected" );
    return;
  }

  // If no data or data change, recreate the field.
  if( error_ ||
      !fHandle_.get_rep() ||
      generation ||
      iwrap_ != iWrap_.get() ||
      jwrap_ != jWrap_.get() ||
      kwrap_ != kWrap_.get() ||
      grid_  != gridData_.get() ) {
 
    error_ = false;

    iwrap_ = iWrap_.get();
    jwrap_ = jWrap_.get();
    kwrap_ = kWrap_.get();

    grid_ = gridData_.get();

    // For flexibility allow the user not to enter a dataset for the data. 
    // This can be merged later with ManageData or when animating.
    if( nHandles.size() < 2 )
      data_ = -1;
    else
      data_ = grid_ ? 0 : 1;

    int idim;
    int jdim;
    int kdim;

    vector<unsigned int> mdims;

    nHandle = nHandles[grid_];

    if( nHandle->nrrd->axis[2].size != 3 ) {
      error( "Grid dataset does not contain 3D points." );
      error_ = true;
      return;
    }
    
    // Create the mesh.
    if( nHandle->nrrd->dim == 5 ) {
      // 3D StructHexVol
      idim = nHandle->nrrd->axis[1].size;
      jdim = nHandle->nrrd->axis[2].size;
      kdim = nHandle->nrrd->axis[3].size;
      
    } else if( nHandle->nrrd->dim == 4 ) {
      // 2D StructQuadSurf
      idim = nHandle->nrrd->axis[1].size;
      jdim = nHandle->nrrd->axis[2].size;
      kdim = 1;
    } else if( nHandle->nrrd->dim == 3 ) {
      // 1D StructCurve
      idim = nHandle->nrrd->axis[1].size;
      jdim = 1;
      kdim = 1;
    } else {
      error( "Grid dimensions do not make sense." );
      error_ = true;
      return;
    }

    // Create the mesh.
    if( idim > 1 && jdim > 1 && kdim > 1 ) {
      // 3D StructHexVol
      mHandle = scinew
	StructHexVolMesh( idim+iwrap_, jdim+jwrap_, kdim+kwrap_ );
 
      mdims.push_back( idim );
      mdims.push_back( jdim );
      mdims.push_back( kdim );

    } else if( idim >  1 && jdim >  1 && kdim == 1 ) {
      // 2D StructQuadSurf
      mHandle = scinew StructQuadSurfMesh(idim+iwrap_, jdim+jwrap_ );

      mdims.push_back( idim );
      mdims.push_back( jdim );
      kwrap_ = 0;

    } else if( idim >  1 && jdim == 1 && kdim  > 1 ) {
      // 2D StructQuadSurf
      mHandle = scinew StructQuadSurfMesh(idim+iwrap_, kdim+kwrap_ );

      mdims.push_back( idim );
      mdims.push_back( kdim );
      jwrap_ = 0;

    } else if( idim == 1 && jdim >  1 && kdim  > 1 ) {
      // 2D StructQuadSurf
      mHandle = scinew StructQuadSurfMesh(jdim+jwrap_, kdim+kwrap_ );

      mdims.push_back( jdim );
      mdims.push_back( kdim );
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
      PPPLNrrdConverterMeshAlgo::get_compile_info(mtd, nHandle->nrrd->type);

    Handle<PPPLNrrdConverterMeshAlgo> algo_mesh;
      
    if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;

    algo_mesh->execute(mHandle, nHandle,
		       idim, jdim, kdim,
		       iwrap_, jwrap_, kwrap_);

    int ndims = mHandle.get_rep()->dimensionality();

    
    // Set the rank of the data Scalar(1), Vector(3), Tensor(6).
    // Assume all of the input nrrd data is scalar with the last axis
    // size being the rank of the data.

    vector<unsigned int> ddims;
    int rank = 0;

    if( data_ != -1 ) {
      nHandle = nHandles[data_];

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

      for( int ic=1; ic<nHandle->nrrd->dim; ic++ ) {

	if( nHandle->nrrd->axis[ic].size != 1 )
	  ddims.push_back( nHandle->nrrd->axis[ic].size );
      }

      if( ddims.size() == mdims.size() ||
	  ddims.size() == mdims.size() + 1 ) {
	
	for( int ic=0; ic<mdims.size(); ic++ ) {
	  if( ddims[ic] != mdims[ic] ) {
	    error( "Data and grid sizes do not match." );
	    error_ = true;
	    return;
	  }
	}

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
	}
      } else {
	error( "Can not determined data rank. Too many axii." );
	error( dataset[0] );
	error_ = true;
	return;
      }
    } else {
      nHandle = NULL;
      rank = 1;
    }

    // Add in the data.
    if( ndims == 3 ) {

      // 3D StructHexVol
      StructHexVolMesh *mesh = (StructHexVolMesh *) mHandle.get_rep();

      if( rank == 1 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructHexVolField< float >(mesh, Field::NODE);
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
	  scinew StructQuadSurfField< float >(mesh, Field::NODE);
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

      // 2D StructCurve
      StructCurveMesh *mesh = (StructCurveMesh *) mHandle.get_rep();

      if( rank == 1 ) {
	// Now after the mesh has been created, create the field.
	fHandle_ =
	  scinew StructCurveField< float >(mesh, Field::NODE);
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
		
    if( data_ != -1 ) {
      const TypeDescription *ftd = fHandle_->get_type_description();
      
      CompileInfoHandle ci =
	PPPLNrrdConverterFieldAlgo::get_compile_info(ftd, nHandle->nrrd->type, rank);
      
      Handle<PPPLNrrdConverterFieldAlgo> algo;

      if (!module_dynamic_compile(ci, algo)) return;

      algo->execute(fHandle_, nHandle,
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
 PPPLNrrdConverter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}



CompileInfoHandle
PPPLNrrdConverterMeshAlgo::get_compile_info(const TypeDescription *ftd,
			       const unsigned int type)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("PPPLNrrdConverterMeshAlgo");
  static const string template_class_name("PPPLNrrdConverterMeshAlgoT");

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
PPPLNrrdConverterFieldAlgo::get_compile_info(const TypeDescription *ftd,
				const unsigned int type,
				int rank)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("PPPLNrrdConverterFieldAlgo");

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



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
 *  PrismNrrdConverter.cc:
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

#include <Packages/Fusion/Dataflow/Modules/Fields/PrismNrrdConverter.h>

#include <sci_defs.h>

namespace Fusion {

using namespace SCIRun;
using namespace SCITeem;

class PSECORESHARE PrismNrrdConverter : public Module {
public:
  PrismNrrdConverter(GuiContext*);

  virtual ~PrismNrrdConverter();

  virtual void execute();
  virtual void error(const std::string& str) { cerr << str << endl; }

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiString datasetsStr_;
  GuiInt pointData_;
  GuiInt connectData_;

  int points_;
  int connect_;
  int data_;

  bool error_;

  vector< int > nGenerations_;

  FieldHandle fHandle_;
};


DECLARE_MAKER(PrismNrrdConverter)
PrismNrrdConverter::PrismNrrdConverter(GuiContext* context)
  : Module("PrismNrrdConverter", context, Source, "Fields", "Fusion"),
    datasetsStr_(context->subVar("datasets")),

    pointData_(context->subVar("points")),
    connectData_(context->subVar("connect")),

    points_(-1),
    connect_(-1),
    data_(-1),

    error_( false )
{
}

PrismNrrdConverter::~PrismNrrdConverter(){
}

void
PrismNrrdConverter::execute(){

  vector< NrrdDataHandle > nHandles;
  NrrdDataHandle nHandle;

  // Assume a range of ports even though only two are needed for the
  // grid and one for the data.
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
      nHandles.push_back( nHandle );
    }
    else if( pi != range.second ) {
      error( "No handle or representation" );
      return;
    }
  }

  if( nHandles.size() == 0 ){
    error( "No handle or representation" );
    return;
  }

  if( nHandles.size() > 3 ) {
    remark( "Too many inputs, ignoring extras." );
  }

  int generation = 0;

  // See if input data has been added or removed.
  if( nGenerations_.size() == 0 ||
      nGenerations_.size() != nHandles.size()) {
    generation = nHandles.size();
  } else {
    // See if any of the input data has changed.
    for( int ic=0; ic<nHandles.size() && ic<nGenerations_.size(); ic++ ) {
      if( nGenerations_[ic] != nHandles[ic]->generation ) {
	++generation;
      }
    }
  }

  // If data change, update the GUI the field if needed.
  if( generation ) {

    nGenerations_.resize( nHandles.size() );

    for( int ic=0; ic++; ic<nHandles.size() ) {
      nGenerations_[ic] = nHandles[ic]->generation;
    }

    string datasetsStr;

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
    }
  
    if( datasetsStr != datasetsStr_.get() ) {
      // Update the dataset names and dims in the GUI.
      ostringstream str;
      str << id << " set_names { " << datasetsStr << "}";
      
      gui->execute(str.str().c_str());
    }
  }

  // Make sure a dataset for the points has been selected.
  if( pointData_.get() < 0 ) {
    warning( "No point dataset selected." );
    return;
  }

  // Make sure a dataset for the connection list has been selected.
  if( connectData_.get() < 0 ) {
    warning( "No connection dataset selected." );
    return;
  }

  // Make sure a dataset for the connection list has been selected.
  if( pointData_.get() == connectData_.get() ) {
    warning( "Point and connection datasets can not be the same." );
    return;
  }

  // If no data or data change, recreate the field.
  if( error_ ||
      !fHandle_.get_rep() ||
      generation ||
      points_  != pointData_.get() ||
      connect_ != connectData_.get() ) {
 
    error_ = false;

    points_  = pointData_.get();
    connect_ = connectData_.get();

    // For flexibility allow the user not to enter a dataset for the data. 
    // This can be merged later with ManageData or when animating.
    if( nHandles.size() < 3 )
      data_ = -1;
    else {
      for( int i=0; i<nHandles.size(); i++ ) {
	if( i != points_ && i != connect_ ) {
	  data_ = i;
	  break;
	}
      }
    }

    vector<unsigned int> mdims;

    NrrdDataHandle pHandle = nHandles[points_];
    NrrdDataHandle cHandle = nHandles[connect_];

    if( pHandle->nrrd->axis[2].size != 3 ) {
      error( "Point data set does not contain 3D points." );
      error_ = true;
      return;
    }
    
    if( cHandle->nrrd->axis[2].size != 6 ) {
      error( "Connection data set does not contain 6 points." );
      error_ = true;
      return;
    }
    
    mdims.push_back( pHandle->nrrd->axis[1].size );

    PrismVolMesh *mesh  = scinew PrismVolMesh();    
    MeshHandle mHandle = mesh;

    const TypeDescription *mtd = mHandle->get_type_description();
      
    CompileInfoHandle ci_mesh =
      PrismNrrdConverterMeshAlgo::get_compile_info(mtd,
						   pHandle->nrrd->type,
						   cHandle->nrrd->type);

    Handle<PrismNrrdConverterMeshAlgo> algo_mesh;
      
    if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;

    algo_mesh->execute(mHandle, pHandle, cHandle);

    
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

      for( int ic=1; ic<nHandle->nrrd->dim; ic++ )
	ddims.push_back( nHandle->nrrd->axis[ic].size );

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

	  if( dataset[0].find( ":Scalar" ) != std::string::npos ) {
	    rank = ddims[mdims.size()];
	  } else {
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


    if( rank == 1 ) {
      // Now after the mesh has been created, create the field.
      fHandle_ = scinew PrismVolField< float >(mesh, Field::NODE);
    } else if( rank == 3 ) {
      // Now after the mesh has been created, create the field.
      fHandle_ = scinew PrismVolField< Vector >(mesh, Field::NODE);
    } else if( rank == 6 ) {
      // Now after the mesh has been created, create the field.
      fHandle_ = scinew PrismVolField< Tensor >(mesh, Field::NODE);
    }
		
    if( data_ != -1 ) {
      const TypeDescription *ftd = fHandle_->get_type_description();
      
      CompileInfoHandle ci =
	PrismNrrdConverterFieldAlgo::get_compile_info(ftd, nHandle->nrrd->type, rank);
      
      Handle<PrismNrrdConverterFieldAlgo> algo;

      if (!module_dynamic_compile(ci, algo)) return;

      algo->execute(fHandle_, nHandle);
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
 PrismNrrdConverter::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}



CompileInfoHandle
PrismNrrdConverterMeshAlgo::get_compile_info(const TypeDescription *ftd,
					     const unsigned int ptype,
					     const unsigned int ctype)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("PrismNrrdConverterMeshAlgo");
  static const string template_class_name("PrismNrrdConverterMeshAlgoT");

  string pTypeStr,  cTypeStr;
  string pTypeName, cTypeName;

  string typeStr;
  string typeName;

  for( int i=0; i<2; i++ ) {
    unsigned int type;

    if( i == 0 ) type = ptype;
    else         type = ctype;

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

    if( i == 0 ) { pTypeStr = typeStr; pTypeName = typeName; }
    else         { cTypeStr = typeStr; cTypeName = typeName; }
  }

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       pTypeName + "." + cTypeName + ".",
                       base_class_name, 
                       template_class_name,
                       ftd->get_name() + ", " + pTypeStr + ", " + cTypeStr );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  ftd->fill_compile_info(rval);
  return rval;
}

CompileInfoHandle
PrismNrrdConverterFieldAlgo::get_compile_info(const TypeDescription *ftd,
				const unsigned int type,
				int rank)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("PrismNrrdConverterFieldAlgo");

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



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

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Packages/Fusion/Dataflow/Modules/Fields/NIMRODConverter.h>

namespace Fusion {

using namespace SCIRun;

class NIMRODConverter : public Module {
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
  vector< GuiInt* > gModes_;

  int nmodes_;
  vector< int > modes_;

  unsigned int conversion_;

  GuiInt allowUnrolling_;
  GuiInt unRolling_;
  int unrolling_;

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
    nmodes_(0),
    conversion_(NONE),
    allowUnrolling_(context->subVar("allowUnrolling")),
    unRolling_(context->subVar("unrolling")),
    unrolling_(0),
    error_(false)
{
}

NIMRODConverter::~NIMRODConverter(){
}

void
NIMRODConverter::execute(){

  vector< NrrdDataHandle > nHandles;
  NrrdDataHandle nHandle;

  string datasetsStr;

  string nrrdName;
  string property;

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
      // Get the name of the nrrd being worked on.
      if( !nHandle->get_property( "Name", nrrdName ) ||
	  nrrdName == "Unknown" ) {
	error( "Can not find the name of the nrrd or it is unknown." );
	return;
      }

      // Get the source of the nrrd being worked on.
      if( !nHandle->get_property( "Source", property ) ||
	  (property != "HDF5" && property != "MDSPlus") ) {
	error( "Can not find the source of the nrrd or it is unknown." );
	return;
      }

      nHandles.push_back( nHandle );

    } else if( pi != range.second ) {
      error( "No handle or representation." );
      return;
    }
  }

  for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {

    nHandles[ic]->get_property( "Name", nrrdName );

    // Save the name of the dataset.
    if( nHandles.size() == 1 )
      datasetsStr.append( nrrdName );
    else
      datasetsStr.append( "{" + nrrdName + "} " );
  }

  if( datasetsStr != datasetsStr_.get() ) {
    // Update the dataset names and dims in the GUI.
    ostringstream str;
    str << id << " set_names " << " {" << datasetsStr << "}";
    
    gui->execute(str.str().c_str());
  }

  if( nHandles.size() != 1 && nHandles.size() != 2 &&
      nHandles.size() != 3 && nHandles.size() != 4 &&
      nHandles.size() != 8 ){
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

  // If data change, update the GUI the field if needed.
  if( generation ) {
    conversion_ = NONE;

    remark( "Found new data ... updating." );

    nGenerations_.resize( nHandles.size() );
    mesh_.resize(4);
    mesh_[0] = mesh_[1] = mesh_[2] = mesh_[3] = -1;

    for( unsigned int ic=0; ic<nHandles.size(); ic++ )
      nGenerations_[ic] = nHandles[ic]->generation;

    // Get each of the dataset names for the GUI.
    for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {

      nHandle = nHandles[ic];

      // Get the name of the nrrd being worked on.
      nHandle->get_property( "Name", nrrdName );

      if( nHandle->get_property( "Topology", property ) ) {

	// Structured mesh.
	if( property.find( "Structured" ) != string::npos ) {

	  if( nHandle->get_property( "Coordinate System", property ) ) {

	    // Special Case - NIMROD data which has multiple components.
	    if( property.find("Cylindrical - NIMROD") != string::npos ) {

	      // Sort the components components.
	      if( nrrdName.find( "R:Scalar" ) != string::npos &&
		  nHandle->nrrd->dim == 2 ) {
		conversion_ = MESH;
		mesh_[R] = ic;
	      } else if( nrrdName.find( "Z:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 2 ) {
		conversion_ = MESH;
		mesh_[Z] = ic; 
	      } else if( nrrdName.find( "PHI:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 1 ) {
		mesh_[PHI] = ic;
	      } else if( nrrdName.find( "K:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 1 ) {
		conversion_ = PERTURBED;
		mesh_[K] = ic;
	      } else {
		error( nrrdName + " is unknown NIMROD mesh data." );
		error_ = true;
		return;
	      }
	    } else {
	      error( nrrdName + property + " is an unsupported coordinate system." );
	      error_ = true;
	      return;
	    }
	  } else {
	    error( nrrdName + "No coordinate system found." );
	    error_ = true;
	    return;
	  }
	} else {
	  error( nrrdName + property + "is an unsupported topology." );
	  error_ = true;
	  return;
	}
      } else if( nHandle->get_property( "DataSpace", property ) ) {

	if( property.find( "REALSPACE" ) != string::npos ) {

	  if( data_.size() == 0 ) {
	    data_.resize(3);
	    data_[0] = data_[1] = data_[2] = -1;
	  }

	  if( nrrdName.find( "R:Scalar" ) != string::npos &&
	      nHandle->nrrd->dim == 3 ) {
	    conversion_ = REALSPACE;
	    data_[0] = ic;
	  } else if( nrrdName.find( "Z:Scalar" ) != string::npos && 
		     nHandle->nrrd->dim == 3 ) {
	    conversion_ = REALSPACE;
	    data_[1] = ic; 
	  } else if( nrrdName.find( "PHI:Scalar" ) != string::npos && 
		     nHandle->nrrd->dim == 3 ) {
	    conversion_ = REALSPACE;
	    data_[2] = ic;
	  } else if( nrrdName.find( ":Scalar" ) != string::npos && 
		     nHandle->nrrd->dim == 3 ) {
	    conversion_ = SCALAR;
	    data_[0] = ic;
	  } else if( (nrrdName.find( "R-Z-PHI:Vector" ) != string::npos || 
		      nrrdName.find( "PHI-R-Z:Vector" ) != string::npos ) && 
		     nHandle->nrrd->dim == 4 ) {
	    conversion_ = REALSPACE;
	    data_.resize(1);
	    data_[0] = ic;
	  } else {
	    error( nrrdName + " is unknown NIMROD REALSPACE node data." );
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
	      else {
		error( nrrdName + property + " Unsupported Data Subspace." );
		error_ = true;
		return;
	      }
	      
	      int index = 0;
	      if( nrrdName.find( "R:Scalar" ) != string::npos &&
		  nHandle->nrrd->dim == 3 ) {
		conversion_ = PERTURBED;
		index = 0 + 3 * offset;
	      } else if( nrrdName.find( "Z:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 3 ) {
		conversion_ = PERTURBED;
		index = 1 + 3 * offset;
	      } else if( nrrdName.find( "PHI:Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 3 ) {
		conversion_ = PERTURBED;
		index = 2 + 3 * offset;
	      } else if( (nrrdName.find( "R-Z-PHI:Vector" ) != string::npos || 
			  nrrdName.find( "PHI-R-Z:Vector" ) != string::npos ) && 
			 nHandle->nrrd->dim == 4 ) {
		conversion_ = PERTURBED;
		data_.resize(2);
		index = 0 + 1 * offset;
	      } else if( nrrdName.find( ":Scalar" ) != string::npos && 
			 nHandle->nrrd->dim == 3 ){ // Scalar data
		conversion_ = PERTURBED;
		data_.resize(2);
		index = 0 + 1 * offset;
	      } else {
		if( offset == 0 )
		  error( nrrdName + " is unknown NIMROD PERTURBED REAL node data." );
		else if( offset == 1 )
		  error( nrrdName + " is unknown NIMROD PERTURBED IMAGINARY node data." );
		error_ = true;
		return;
	      }

	      data_[index] = ic;

	    } else {
	      error( nrrdName + property + " Unsupported Data SubSpace." );
	      error_ = true;
	      return;
	    }
	  } else {
	    error( nrrdName + " No Data SubSpace property." );
	    error_ = true;
	    return;
	  }
	} else {	
	  error( nrrdName + property + " Unsupported Data Space." );
	  error_ = true;
	  return;
	}
      } else {
	error( nrrdName + " No DataSpace property." );
	error_ = true;
	return;
      }
    }
  }

  unsigned int i = 0;
  while( i<data_.size() && data_[i] != -1 )
    i++;
  
  if( conversion_ & MESH ) {
    if( mesh_[R] == -1 || mesh_[Z] == -1 || mesh_[PHI] == -1 ) {
      error( "Not enough mesh data for the mesh conversion." );
      error_ = true;
      return;
    }
  } else if ( conversion_ & REALSPACE ) {
    if( mesh_[PHI] == -1 || i != data_.size() ) {
      error( "Not enough data for the realspace conversion." );
      error_ = true;
      return;
    }
  } else if ( conversion_ & PERTURBED ) {
    if( mesh_[PHI] == -1 || mesh_[K] == -1 || i != data_.size() ) {
      error( "Not enough data for the perturbed conversion." );
      error_ = true;
      return;
    }
  }

  if( (int) (conversion_ & MESH) != allowUnrolling_.get() )
  {
    ostringstream str;
    str << id << " set_unrolling " << (conversion_ & MESH);
    
    gui->execute(str.str().c_str());

    if( conversion_ & MESH ) {
      warning( "Select the mesh rolling for the calculation" );
      error_ = true; // Not really an error but it so it will execute.
      return;
    }
  }

  if( (conversion_ & PERTURBED) &&
      nHandles[mesh_[K]]->get_property( "Coordinate System", property ) &&
      property.find("Cylindrical - NIMROD") != string::npos )
    nmodes_ = nHandles[mesh_[K]]->nrrd->axis[0].size; // Modes
  else
    nmodes_ = 0;

  int nmodes = gModes_.size();

  // Remove the GUI entries that are not needed.
  for( int ic=nmodes-1; ic>nmodes_; ic-- ) {
    delete( gModes_[ic] );
    gModes_.pop_back();
    modes_.pop_back();
  }

  if( nmodes_ > 0 ) {
    // Add new GUI entries that are needed.
    for( int ic=nmodes; ic<=nmodes_; ic++ ) {
      char idx[24];
      
      sprintf( idx, "mode-%d", ic );
      gModes_.push_back(new GuiInt(ctx->subVar(idx)) );
      
      modes_.push_back(0);
    }
  }

  if( nModes_.get() != nmodes_ ) {

    // Update the modes in the GUI
    ostringstream str;
    str << id << " set_modes " << nmodes_ << " 1";

    gui->execute(str.str().c_str());
    
    if( conversion_ & PERTURBED ) {
      warning( "Select the mode for the calculation" );
      error_ = true; // Not really an error but it so it will execute.
      return;
    }
  }


  bool updateMode = false;

  if( nmodes_ > 0 ) {
    bool haveMode = false;

    for( int ic=0; ic<=nmodes_; ic++ ) {
      gModes_[ic]->reset();
      if( modes_[ic] != gModes_[ic]->get() ) {
	modes_[ic] = gModes_[ic]->get();
	updateMode = true;
      }

      if( !haveMode ) haveMode = (modes_[ic] ? true : false );
    }

    if( !haveMode ) {
      warning( "Select the mode for the calculation" );
      error_ = true; // Not really an error but it so it will execute.
      return;
    }
  }

  bool updateRoll = false;

  if( unrolling_ != unRolling_.get() ) {
    unrolling_ = unRolling_.get();
    updateRoll = true;
  }

  // If no data or data change, recreate the field.
  if( error_ ||
      updateMode ||
      updateRoll ||
      !nHandle_.get_rep() ||
      generation ) {
    
    error_ = false;

    string convertStr;
    unsigned int ntype;
    
    if( conversion_ & MESH ) {
      ntype = nHandles[mesh_[PHI]]->nrrd->type;

      nHandles[mesh_[PHI]]->get_property( "Coordinate System", property );
      
      if( nHandles[mesh_[R]]->nrrd->axis[0].size != 
	  nHandles[mesh_[Z]]->nrrd->axis[0].size ||
	  nHandles[mesh_[R]]->nrrd->axis[1].size != 
	  nHandles[mesh_[Z]]->nrrd->axis[1].size ) {
	error( "R-Z Mesh dimension mismatch." );
	error_ = true;
	return;
      }
      
      modes_.resize(1);
      modes_[0] = unrolling_;

      convertStr = "Mesh";
      
    } else if( conversion_ & SCALAR ) {
      ntype = nHandles[data_[0]]->nrrd->type;

      convertStr = "Scalar";
      
    } else if( conversion_ & REALSPACE ) {
      ntype = nHandles[mesh_[PHI]]->nrrd->type;
      
      nHandles[mesh_[PHI]]->get_property( "Coordinate System", property );

      if( property.find("Cylindrical - NIMROD") != string::npos ) {
	if(  data_.size() == 1 ) {
	  if( nHandles[mesh_[PHI]]->nrrd->axis[0].size !=
	      nHandles[data_[  0]]->nrrd->axis[3].size ) {
	    error( "Phi Mesh - Data dimension mismatch." );
	    error_ = true;
	    return;
	  }

	} else if( data_.size() == 3 ) {
	  if( property.find("Cylindrical - NIMROD") != string::npos ) {
	    if( nHandles[mesh_[PHI]]->nrrd->axis[0].size !=
		nHandles[data_[  0]]->nrrd->axis[3].size ) {
	      error( "Phi Mesh - Data dimension mismatch." );
	      error_ = true;
	      return;
	    }

	    for( unsigned int ic=1; ic<data_.size(); ic++ ) {
	      for( int jc=1; jc<nHandles[data_[0]]->nrrd->dim; jc++ ) {
		if( nHandles[data_[ 0]]->nrrd->axis[jc].size !=
		    nHandles[data_[ic]]->nrrd->axis[jc].size ) {
		  error( "Data dimension mismatch." );
		  error_ = true;
		  return;
		}
	      }
	    }
	  }
	}
      }

      convertStr = "RealSpace";

    } else if( conversion_ & PERTURBED ) {
      ntype = nHandles[mesh_[PHI]]->nrrd->type;

      nHandles[mesh_[PHI]]->get_property( "Coordinate System", property );

      if( property.find("Cylindrical - NIMROD") != string::npos ) {

	int cc = nHandles[data_[0]]->nrrd->dim;

	if( nHandles[mesh_[K]]->nrrd->axis[0].size !=
	    nHandles[data_[0]]->nrrd->axis[cc-1].size ) {
	  error( "Perturbed Mode Mesh - Data dimension mismatch." );
	  error_ = true;
	  return;
	}

	for( unsigned int ic=1; ic<data_.size(); ic++ ) {
	  for( int jc=0; jc<cc; jc++ ) {
	    
	    if( nHandles[data_[ 0]]->nrrd->axis[jc].size !=
		nHandles[data_[ic]]->nrrd->axis[jc].size ) {
	      error( "Data dimension mismatch." );
	      error_ = true;
	      return;
	    }
	  }
	}
      }

      convertStr = "Perturbed";
    }

    if( conversion_ ) {
      remark( "Converting the " + convertStr );
    
      CompileInfoHandle ci_mesh =
	NIMRODConverterAlgo::get_compile_info( convertStr, ntype );
    
      Handle<NIMRODConverterAlgo> algo_mesh;
    

      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) {
	error_ = true;
	return;
      }

      nHandle_ = algo_mesh->execute( nHandles, mesh_, data_, modes_ );
    } else {
      warning( "Nothing to convert." );
      return;
    }

    if( conversion_ & MESH )
      modes_.clear();
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
NIMRODConverterAlgo::get_compile_info( const string converter,
				       const unsigned int ntype )
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string base_class_name( "NIMRODConverterAlgo");
  const string template_class_name( "NIMROD" + converter + "ConverterAlgoT");

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

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
 *  NrrdToField.cc:  Convert a Nrrds to a Field. Incoming Nrrds
 *                    may consist of Data, Points, Connections and
 *                    a Field mesh to associate it with.  
 *
 *  Written by:
 *   Darby Van Uitert
 *   School of Computing
 *   University of Utah
 *   March 2004
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <iostream>

#include <Teem/Dataflow/Modules/Converters/NrrdToField.h>

namespace SCITeem {

using namespace SCIRun;

class NrrdToField : public Module {
public:
  enum {UNKNOWN=0,UNSTRUCTURED=1,STRUCTURED=2,IRREGULAR=4,REGULAR=8};

  NrrdIPort*  ndata_;
  NrrdIPort*  npoints_;
  NrrdIPort*  nconnect_;
  FieldIPort* ifield_;
  FieldOPort* ofield_;

  GuiInt      permute_;
  GuiInt      build_eigens_;
  GuiString   quad_or_tet_;
  GuiString   struct_unstruct_;
  GuiString   datasets_;
  int         data_generation_, points_generation_;
  int         connect_generation_, origfield_generation_;
  bool        has_error_;
  FieldHandle last_field_;

  NrrdToField(GuiContext*);

  virtual ~NrrdToField();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // dataH is a handle to the nrrd containing data.  If there
  // is not data, pass a 0. Only the regular data must have a
  // valid dataH. The nrrd should be a 1D array of data values
  // when scalar, a 3xn 2D array for vector data, and 6/7/9xn
  // 2D array for tensor data depending on the type of tensor.

  // pointsH is a handle to the nrrd containing the points. There
  // must be points for structured or unstructured data. The points
  // nrrd should be a 2D array of nx3 where n is the number of points
  // and 3 is for an x,y,z.

  // connectH is a handle to the nrrd of connections. The connections
  // nrrd should be a 2D array of pxn where n is the number of connections
  // and p is the number of points per element type.

  // build_eigens should be set to 1 if the for a field of vectors,
  // the eigendecomposition should be computed.  if data is scalar
  // or vector, pass a 0 in.

  // permute should be 1 if the data is in k x j x i format (FORTRAN)

  // For unstructured data with 4 points per element, there is ambiguity
  // whether or not it is a tet or quad.  Set to "Auto" to try to 
  // determine based on Nrrd properties or "Tet" or "Quad".  If it
  // in unclear and "Auto" has been selected, it will default to "Tet".

  // In the case of only points , it must be indicated whether it is
  // a Point Cloud or StructCurve.  Pass the value
  // "Auto" to determine based on Nrrd properties, or "StructCurve",
  // or "PointCloud". If "Auto" is selected and it cannot
  // be determined, a PointCloud will be created.

  FieldHandle create_field_from_nrrds(NrrdDataHandle dataH,
				      NrrdDataHandle pointsH,
				      NrrdDataHandle connectH,
				      FieldHandle    origfieldH,
				      int build_eigens,
				      int permute,
				      const string &quad_or_tet,
				      const string &struct_unstruct);
  
};


DECLARE_MAKER(NrrdToField)
NrrdToField::NrrdToField(GuiContext* ctx)
  : Module("NrrdToField", ctx, Source, "Converters", "Teem"),
    permute_(ctx->subVar("permute")),
    build_eigens_(ctx->subVar("build-eigens")),
    quad_or_tet_(ctx->subVar("quad-or-tet")),
    struct_unstruct_(ctx->subVar("struct-unstruct")),
    datasets_(ctx->subVar("datasets")),
    data_generation_(-1), points_generation_(-1),
    connect_generation_(-1), origfield_generation_(-1),
    has_error_(false), last_field_(0)
{
}

NrrdToField::~NrrdToField(){
}



void
 NrrdToField::execute(){
  // Get ports
  ndata_ = (NrrdIPort *)get_iport("Data");
  npoints_ = (NrrdIPort *)get_iport("Points");
  nconnect_ = (NrrdIPort *)get_iport("Connections");
  ifield_ = (FieldIPort *)get_iport("OriginalField");
  ofield_ = (FieldOPort *)get_oport("OutputField");

  NrrdDataHandle dataH;
  NrrdDataHandle pointsH;
  NrrdDataHandle connectH;
  FieldHandle origfieldH;
  
  bool do_execute = false;
  // Determine if we have data, points, connections, etc.
  if (!ndata_->get(dataH)) {
    dataH = 0;
    if (data_generation_ != -1) {
      data_generation_ = -1;
      do_execute = true;
    }
  } else if (!dataH.get_rep()) {
    error("No data in the Data nrrd.");
  } 
  
  if (!npoints_->get(pointsH)) {
    pointsH = 0;
    if (points_generation_ != -1) {
      points_generation_ = -1;
      do_execute = true;
    }
  } else if (!pointsH.get_rep()) {
    error("No data in the Points nrrd.");
  } 
  
  if (!nconnect_->get(connectH)) {
    connectH = 0;
    if (connect_generation_ != -1) {
      connect_generation_ = -1;
      do_execute = true;
    }
  } else if (!connectH.get_rep()) {
    error("No data in the Connections nrrd.");
  } 
  
  if (!ifield_->get(origfieldH)) {
    origfieldH = 0;
    if (origfield_generation_ != -1) {
      origfield_generation_ = -1;
      do_execute = true;
    }
  } else if (!origfieldH.get_rep()) {
    error("No data in the Originating Field.");
  } 
  
  // check the generations to see if we need to re-execute
  if (dataH != 0 && data_generation_ != dataH->generation) {
    data_generation_ = dataH->generation;
    do_execute = true;
  }
  if (pointsH != 0 && points_generation_ != pointsH->generation) {
    points_generation_ = pointsH->generation;
    do_execute = true;
  }
  if (connectH != 0 && connect_generation_ != connectH->generation) {
    connect_generation_ = connectH->generation;
    do_execute = true;
  }
  if (origfieldH != 0 && origfield_generation_ != origfieldH->generation) {
    origfield_generation_ = origfieldH->generation;
    do_execute = true;
  }

  // set the names of the datasets for the UI
  string datasetsStr = "";
  string property;

  if (pointsH != 0) {
    if (pointsH->get_property( "Name", property ) && property != "Unknown")
      datasetsStr.append( "{Points : " + property + "} ");
    else
      datasetsStr.append( "{Points : Unknown} " );
  } else {
    datasetsStr.append( "{Points : (none)} " );
  }
  if (connectH != 0) {
    if (connectH->get_property( "Name", property ) && property != "Unknown")
      datasetsStr.append( "{Connections: " + property + "} ");
    else
      datasetsStr.append( "{Connections : Unknown} " );
  } else {
    datasetsStr.append( "{Connections : (none)} " );
  }
  if (dataH != 0) {
    if (dataH->get_property( "Name", property ) && property != "Unknown")
      datasetsStr.append( "{Data : " + property + "} ");
    else
      datasetsStr.append( "{Data : Unknown} " );
  } else {
    datasetsStr.append( "{Data : (none)} " );
  }
  if (origfieldH != 0) {
    if (origfieldH->get_property( "Name", property ) && property != "Unknown")
      datasetsStr.append( "{Original Field : " + property + "} ");
    else
      datasetsStr.append( "{Original Field : Unknown} " );
  } else {
    datasetsStr.append( "{Original Field : (none)} " );
  }

  datasets_.reset();
  if( datasetsStr != datasets_.get() ) {
    // Update the dataset names and dims in the GUI.
    ostringstream str;
    str << id << " set_names {" << datasetsStr << "}";
    
    gui->execute(str.str().c_str());
  }

  if (has_error_)
    do_execute = true;

  // execute the module
  if (do_execute) {
    last_field_ = create_field_from_nrrds(dataH, pointsH, connectH, origfieldH,
					  build_eigens_.get(), permute_.get(), 
					  quad_or_tet_.get(),
					  struct_unstruct_.get());
    
  } 
  if (last_field_ != 0) {
    has_error_ = false;
    // set the name of the field to be that
    // of data or points, depending on what is defined
    string field_name;
    if (dataH != 0) {
      if (dataH->get_property( "Name", property ) && property != "Unknown")
	field_name = property;
    } else if (pointsH != 0) {
      if (pointsH->get_property( "Name", property ) && property != "Unknown")
	field_name = property;
    } else {
      field_name = "Unknown";
    }
    last_field_->set_property("name", field_name, false);
    ofield_->send(last_field_);  
  }
}

FieldHandle 
NrrdToField::create_field_from_nrrds(NrrdDataHandle dataH,
				     NrrdDataHandle pointsH,
				     NrrdDataHandle connectH,
				     FieldHandle origfieldH,
				     int build_eigens,
				     int permute,
				     const string &quad_or_tet, 
				     const string &struct_unstruct) {
				    
  MeshHandle mHandle;
  int idim = 1, jdim = 1, kdim = 1; // initialize to 1 so it will at least go through i,j,k loops once.
  string property;
  unsigned int topology_ = UNKNOWN;
  unsigned int geometry_ = UNKNOWN;
  FieldHandle ofield_handle = 0;
  

  // Determine if we have data, points, connections, etc.
  bool has_data_ = false, has_points_ = false, has_connect_ = false, 
    has_origfield_ = false;

  if (dataH != 0)
    has_data_ = true;
  if (pointsH != 0)
    has_points_ = true;
  if (connectH != 0)
    has_connect_ = true;
  if (origfieldH != 0)
    has_origfield_ = true;


  // Make sure the points and data coming in make sense.
  if (has_points_ && pointsH->nrrd->axis[0].size != 3) {
    error("Points Nrrd must contain 3D points.");
    error("If the points are 2D use UnuPad to turn in to 3D points.");
    has_error_ = true;
    return 0;
  }

  unsigned int data_rank;

  if (has_data_ && nrrdKindSize( dataH->nrrd->axis[0].kind ) > 1) {
    if (dataH->nrrd->axis[0].size == 3) {
      data_rank = 3;
    } else if (dataH->nrrd->axis[0].size >= 6) {
      data_rank = dataH->nrrd->axis[0].size;
    } else {
      error("Data Nrrd must contain 3D scalars, vectors, or tensors.");
      has_error_ = true;
      return 0;
    }
  } else
    data_rank = 1;

  if( has_points_ && has_data_ ) {
    // Data and points must have the same number of dimensions and
    // the all but the first dimension must have the same size.
    if (!(dataH->nrrd->dim == pointsH->nrrd->dim ||
	  dataH->nrrd->dim == (pointsH->nrrd->dim-1))) {
      error ("Data and Points must have the same dimension for the domain.");
      has_error_ = true;
      return 0;
    }
    for (int d = dataH->nrrd->dim-1, p = pointsH->nrrd->dim-1; p > 0; d--, p--) {
      if (dataH->nrrd->axis[d].size != pointsH->nrrd->axis[p].size) {
	error("Data and Points must have the same size for all domain axes.");
	has_error_ = true;
	return 0;
      }
    }
  }


  ///////////// HAS ORIGINATING FIELD /////////////

  if (has_origfield_) {
    // Attempt to use the mesh provided
    mHandle = origfieldH->mesh();

    // verify the data will "fit" into the mesh passed in
    if (has_data_) {
      const TypeDescription *td = origfieldH->get_type_description();
      CompileInfoHandle ci = NrrdToFieldTestMeshAlgo::get_compile_info(td);
      Handle<NrrdToFieldTestMeshAlgo> algo;
      if ((module_dynamic_compile(ci, algo)) && 
	  (algo->execute(origfieldH, dataH, ofield_handle, data_rank))) {
	remark("Creating a field from original mesh in input nrrd");
	  
	// Now create field based on the mesh created above and send it
	const TypeDescription *mtd = mHandle->get_type_description();
	  
	string fname = mtd->get_name();
	string::size_type pos = fname.find( "Mesh" );
	fname.replace( pos, 4, "Field" );
	  
	CompileInfoHandle ci;
	  
	ci = NrrdToFieldFieldAlgo::get_compile_info(mtd,
						    fname,
						    dataH->nrrd->type, 
						    data_rank);
	  
	Handle<NrrdToFieldFieldAlgo> algo;
	if (!module_dynamic_compile(ci, algo)) return 0;
	  
	if ((nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) && build_eigens) {
	  warning("Attempting to build eigendecomposition of Tensors with non symmetric tensor");
	}
	  
	ofield_handle = algo->execute( mHandle, dataH, build_eigens);
	  
	return ofield_handle;
      } else {
	warning("Warning: Field passed in but not used because sizes didn't match.  Generating mesh from scratch. Try inserting a ChangeFieldDataAt module above NrrdToField and setting the data location.");
      }
    } else if (has_points_) {
      // Now create field based on the mesh created above and send it
      const TypeDescription *mtd = mHandle->get_type_description();
      
      string fname = mtd->get_name();
      string::size_type pos = fname.find( "Mesh" );
      fname.replace( pos, 4, "Field" );
      
      CompileInfoHandle ci;
      
      ci = NrrdToFieldFieldAlgo::get_compile_info(mtd,
						  fname,
						  pointsH->nrrd->type, 
						  data_rank);
      
      Handle<NrrdToFieldFieldAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return 0;
      
      // DARBY - change this replace all of the current points to the new points.
      ofield_handle = algo->execute( mHandle, dataH, build_eigens);
      
      return ofield_handle;
    } else {
      error("Have original field but no points or data.");
      has_error_ = true;
      return 0;
    }
  }
  

  /////////////////// DETERMINE MESH BASED ON INPUTS AND GUI INFO ///////////////////

  // No field given, does it have points and connections and possibly data?
  if (has_points_ && has_connect_) {

    Nrrd* connect = connectH->nrrd;

    if (connect->dim != 1 && connect->dim != 2) {
      error("Connections Nrrd must be two dimensional (number of points in each connection by the number of elements)");
      has_error_ = true;
      return 0;
    }

    int which = 0; // which index contains p
    int pts = connect->axis[which].size;
      
    if (pts != 2 && pts != 3 && pts != 4 && pts != 6 && pts != 8) {

      if( connect->dim == 1 ) {
	error("Connections nrrd must have one axis with size 2, 3, 4, 6, or 8.");
	has_error_ = true;
	return 0;

      } else {
	warning("Connections nrrd might not be p x n where p is the number of points in each connection and n is the number of elements.");
	warning("Assuming n x p.");

	which = 1;
	pts = connect->axis[which].size;
      
	if (pts != 2 && pts != 3 && pts != 4 && pts != 6 && pts != 8) {

	  error("Connections nrrd must have one axis with size 2, 3, 4, 6, or 8.");
	  has_error_ = true;
	  return 0;
	}
      }
    }

    // If there is an elem type property make sure it matches what is 
    // found automatically.
    string elem_type = "Auto";
    int nconnections = 0;

    if ( connectH->get_property( "Elem Type",property )) {
      if( property.find( "Curve" ) != string::npos ) {
	elem_type = "Curve";
	nconnections = 2;
      } else if( property.find( "Tri" ) != string::npos ) {
	elem_type = "Tri";
	nconnections = 3;
      } else if( property.find( "Tet" ) != string::npos ) {
	elem_type = "Tet";
	nconnections = 4;
      } else if( property.find( "Quad" ) != string::npos ) {
	elem_type = "Quad";
	nconnections = 4;
      } else if( property.find( "Prism" ) != string::npos ) {
	elem_type = "Prism";
	nconnections = 6;
      } else if( property.find( "Hex" ) != string::npos ) {
	elem_type = "Hex";
	nconnections = 8;
      }
    }

    if( nconnections == 0 ) {
      if (pointsH->get_property( "Elem Type",property )) {
	warning("Elem Type defined in Points nrrd instead of Connectivity nrrd.");
	if( property.find( "Curve" ) != string::npos ) {
	  elem_type = "Curve";
	  nconnections = 2;
	} else if( property.find( "Tri" ) != string::npos ) {
	  elem_type = "Tri";
	  nconnections = 3;
	} else if( property.find( "Tet" ) != string::npos ) {
	  elem_type = "Tet";
	  nconnections = 4;
	} else if( property.find( "Quad" ) != string::npos ) {
	  elem_type = "Quad";
	  nconnections = 4;
	} else if( property.find( "Prism" ) != string::npos ) {
	  elem_type = "Prism";
	  nconnections = 6;
	} else if( property.find( "Hex" ) != string::npos ) {
	  elem_type = "Hex";
	  nconnections = 8;
	}
      }
    }

    if( nconnections == 0 && has_data_ ) {
      if (dataH->get_property( "Elem Type",property )) {
	warning("Elem Type defined in Data nrrd instead of Connectivity nrrd.");
	if( property.find( "Curve" ) != string::npos ) {
	  elem_type = "Curve";
	  nconnections = 2;
	} else if( property.find( "Tri" ) != string::npos ) {
	  elem_type = "Tri";
	  nconnections = 3;
	} else if( property.find( "Tet" ) != string::npos ) {
	  elem_type = "Tet";
	  nconnections = 4;
	} else if( property.find( "Quad" ) != string::npos ) {
	  elem_type = "Quad";
	  nconnections = 4;
	} else if( property.find( "Prism" ) != string::npos ) {
	  elem_type = "Prism";
	  nconnections = 6;
	} else if( property.find( "Hex" ) != string::npos ) {
	  elem_type = "Hex";
	  nconnections = 8;
	}
      }
    }

    if( nconnections && nconnections != pts ) {
      warning("The elem type properties and the number of connections found do not match.");
    }

    int connectivity = 0;
    topology_ = UNSTRUCTURED;
    geometry_ = IRREGULAR;

    switch (pts) {
    case 2:
      // 2 -> curve
      mHandle = scinew CurveMesh();
      connectivity = 2;
      break;
    case 3:
      // 3 -> tri
      mHandle = scinew TriSurfMesh();
      connectivity = 3;
      break;
    case 4:
      // 4 -> quad/tet (ask which if this case)
      if (quad_or_tet == "Quad") {
	mHandle = scinew QuadSurfMesh();
	connectivity = 4;
      } else  if (quad_or_tet == "Tet") {
	mHandle = scinew TetVolMesh();
	connectivity = 4;
      } else if (elem_type == "Tet") {
	mHandle = scinew TetVolMesh();
	connectivity = 4;	    
      } else if (elem_type == "Quad") {
	mHandle = scinew QuadSurfMesh();
	connectivity = 4;
      } else {
	error("Auto detection of Elem Type using properties failed.");
	error("Connections Nrrd indicates 4 points per connection. Please indicate whether a Tet or Quad in UI.");
	has_error_ = true;
	return 0;
      }
      break;
    case 6:
      // 6 -> prism
      mHandle = scinew PrismVolMesh();
      connectivity = 6;
      break;
    case 8:
      // 8 -> hex
      mHandle = scinew HexVolMesh();
      connectivity = 8;
      break;
    default:
      error("Connections Nrrd must contain 2, 3, 4, 6, or 8 points per connection.");
      has_error_ = true;
      return 0;
    }
      
    // create mesh
    const TypeDescription *mtd = mHandle->get_type_description();

    remark( "Creating an unstructured " + mtd->get_name() );
      
    CompileInfoHandle ci_mesh =
      NrrdToFieldMeshAlgo::get_compile_info("Unstructured",
					    mtd,
					    pointsH->nrrd->type,
					    connectH->nrrd->type);
      
    Handle<UnstructuredNrrdToFieldMeshAlgo> algo_mesh;
      
    if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return 0;
      
    //algo_mesh->execute(mHandle, pointsH, connectH, connectivity, data_size);
    algo_mesh->execute(mHandle, pointsH, connectH, connectivity, which);
    



  } else if (has_points_) { // does it have points and possibly data?

    Nrrd* points = pointsH->nrrd;

    geometry_ = IRREGULAR;

    switch (pointsH->nrrd->dim) {
    case 1:
      mHandle = scinew PointCloudMesh();
      topology_ = UNSTRUCTURED;
      break;

    case 2:
      // data 1D ask if point cloud or structcurvemesh
      if (struct_unstruct == "PointCloud") {
	mHandle = scinew PointCloudMesh();
	topology_ = UNSTRUCTURED;
      } else if (struct_unstruct == "StructCurve") {
	topology_ = STRUCTURED;
	mHandle = scinew StructCurveMesh( points->axis[1].size );
	idim = points->axis[1].size;
      } else {
	// Try to figure out based on properties of the points
	if (pointsH->get_property( "Topology" , property)) {
	  if( property.find( "Unstructured" ) != string::npos ) {
	    topology_ = UNSTRUCTURED;
	    mHandle = scinew PointCloudMesh();
	  } else if( property.find( "Structured" ) != string::npos ) {
	    topology_ = STRUCTURED;
	    mHandle = scinew StructCurveMesh( points->axis[1].size );
	    idim = points->axis[1].size;
	  }
	}
	
	if( topology_ == UNKNOWN ) {
	  if (pointsH->get_property( "Elem Type", property)) { 
	    if ( property.find( "Curve") != string::npos ) {
	      topology_ = UNSTRUCTURED;
	      mHandle = scinew PointCloudMesh();
	    } else if ( property.find( "Curve") != string::npos ) {
	      topology_ = STRUCTURED;
	      mHandle = scinew StructCurveMesh( points->axis[1].size );
	      idim = points->axis[1].size;
	    }
	  }
	}

	if( has_data_ && topology_ == UNKNOWN) {
	  // Try to figure out based on properties of the data
	  if (dataH->get_property( "Topology" , property)) {
	    if( property.find( "Unstructured" ) != string::npos ) {
	      topology_ = UNSTRUCTURED;
	      mHandle = scinew PointCloudMesh();
	    } else if( property.find( "Structured" ) != string::npos ) {
	      topology_ = STRUCTURED;
	      mHandle = scinew StructCurveMesh( points->axis[1].size );
	      idim = points->axis[1].size;
	    }
	  }
	
	  if( topology_ == UNKNOWN ) {
	    if (dataH->get_property( "Elem Type", property)) { 
	      if ( property.find( "Curve") != string::npos ) {
		topology_ = UNSTRUCTURED;
		mHandle = scinew PointCloudMesh();
	      } else if ( property.find( "Curve") != string::npos ) {
		topology_ = STRUCTURED;
		mHandle = scinew StructCurveMesh( points->axis[1].size );
		idim = points->axis[1].size;
	      }
	    }
	  }
	}

	if( topology_ == UNKNOWN ) {
	  warning("Unable to determine if creating Point Cloud or Struct Curve. Defaulting to Point Cloud");
	  mHandle = scinew PointCloudMesh();
	  topology_ = UNSTRUCTURED;
	}
      }
      break;

    case 3:
      topology_ = STRUCTURED;
      // data 2D -> structquad
      mHandle = scinew StructQuadSurfMesh(points->axis[1].size,
					  points->axis[2].size);
      idim = points->axis[1].size;
      jdim = points->axis[2].size;
      //kdim = 1
      break;

    case 4:
      topology_ = STRUCTURED;
      // data 3D -> structhexvol
      mHandle = scinew StructHexVolMesh(points->axis[1].size,
					points->axis[2].size,
					points->axis[3].size);
      idim = points->axis[1].size;
      jdim = points->axis[2].size;
      kdim = points->axis[3].size;
      break;

    default:
      error("Incorrect dimensions for Data Nrrd");
      has_error_ = true;
      return 0;
    }

    // create mesh
    if (topology_ == UNSTRUCTURED) {
      // This would only get here for point clouds with no connectivity
      const TypeDescription *mtd = mHandle->get_type_description();

      remark( "Creating an unstructured " + mtd->get_name() );

      CompileInfoHandle ci_mesh = 0;
      if (has_data_) {
	ci_mesh = NrrdToFieldMeshAlgo::get_compile_info("Unstructured",
					      mtd,
					      pointsH->nrrd->type,
					      dataH->nrrd->type);
      } else {
	ci_mesh = NrrdToFieldMeshAlgo::get_compile_info("Unstructured",
					      mtd,
					      pointsH->nrrd->type,
					      pointsH->nrrd->type);
      }
	  
      Handle<UnstructuredNrrdToFieldMeshAlgo> algo_mesh;
	  
      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return 0;
	  
      //algo_mesh->execute(mHandle, pointsH, connectH, connectivity, data_size);
      int connectivity = 0;
      int which = 0;
      algo_mesh->execute(mHandle, pointsH, connectH, connectivity, which);

    } else {
      const TypeDescription *mtd = mHandle->get_type_description();
	  
      remark( "Creating a structured " + mtd->get_name() );
	  
      CompileInfoHandle ci_mesh =
	NrrdToFieldMeshAlgo::get_compile_info( "Structured",
					       mtd,
					       pointsH->nrrd->type,
					       pointsH->nrrd->type);
	  
      Handle<StructuredNrrdToFieldMeshAlgo> algo_mesh;
	  
      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return 0;
	  
      algo_mesh->execute(mHandle, pointsH, idim, jdim, kdim );
    }


  } else if( has_data_ ) { // No points just data.
    Nrrd* data = dataH->nrrd;

    vector<double> min_pt;
    vector<double> sp;
    int data_center = nrrdCenterUnknown;
    int domain_axis = 0;
    int data_off;

    // must have at 3 points for a point and
    // the min defaults to (0,0,0)
    min_pt.push_back(0.0);
    min_pt.push_back(0.0);
    min_pt.push_back(0.0);

    if( data_rank == 1 )
      data_off = 0;
    else 
      data_off = 1;

    // since there is a mutual exclusion between using
    // the min, max and spacing and the spaceDirection vectors
    // verify that this is a valid nrrd
    if (nrrdCheck(data)) {
      error("Error: Invalid NRRD.  Cannot compute transform.");
      return false;
    }

    // initialize the transform to the identity matrix
    double trans[16];
    trans[0] = 1;  trans[1] = 0;  trans[2] = 0;  trans[3] = 0;
    trans[4] = 0;  trans[5] = 1;  trans[6] = 0;  trans[7] = 0;
    trans[8] = 0;  trans[9] = 0;  trans[10] = 1;  trans[11] = 0;
    trans[12] = 0;  trans[13] = 0;  trans[14] = 0;  trans[15] = 1;

    // handle the spaceOrigin first because it might contain
    // more elements than the nrrd dimension
    for (int i=0; i<dataH->nrrd->spaceDim; i++) {
      double min = 0.0;

      if (airExists(data->spaceOrigin[i])) 
	min = data->spaceOrigin[i];

      if ((int)min_pt.size() == i)
	min_pt.push_back( min );
      else
	min_pt[i] = min;
    }

    // for each axis, determine the min/max/spacing or 
    // spaceDirection vector
    for (int i=data_off; i<dataH->nrrd->dim; i++, domain_axis++) {
      // the origin may have all ready been set if it was coming
      // from the spaceOrigin, otherwise use the min points
      // we are guaranteed that both the spaceOrigin and min
      // will not be set because we did a nrrdCheck
      if (airExists(data->axis[i].min)) {
	if ((int)min_pt.size() == domain_axis)
	  min_pt.push_back( data->axis[i].min );
	else
	  min_pt[ domain_axis ] = data->axis[i].min;
      }

      // get the spacing either from the spaceDirection vector
      // via the nrrdSpacingCalculate call or the axis
      // spacing field -- NOTE that only one of these
      // should be set for each axis in order to have a valid nrrd
      double spac = 0;
      double *space_vec = new double[NRRD_SPACE_DIM_MAX];

      int result = nrrdSpacingCalculate(data, i, &spac, space_vec);

      if (result == nrrdSpacingStatusDirection && airExists(spac)) {
	sp.push_back(spac);

	// set rotation as columns in transform
	if (domain_axis < 3) {
	  trans[domain_axis] = data->axis[i].spaceDirection[0];
	  
	  if (data->spaceDim > 1) 
	    trans[domain_axis+4] = data->axis[i].spaceDirection[1];
	  
	  if (data->spaceDim > 2) 
	    trans[domain_axis+8] = data->axis[i].spaceDirection[2];
	}
      }
      else if (airExists(data->axis[i].spacing)) {
	sp.push_back(data->axis[i].spacing);
      }
      else 
	sp.push_back(1.0);

      if (data->axis[i].center != nrrdCenterUnknown) 
	data_center = data->axis[i].center;

      delete space_vec;
    }

    // Set origin/min to be the 4th column of the transform 
    trans[3]  = min_pt[0];
    trans[7]  = min_pt[1];
    trans[11] = min_pt[2];

    // we have axis aligned data if all values are 0 
    // except on the diagonal
    bool axis_aligned = true;
    if( (Abs(trans[1] - 0.0) > 0.0001) ||
	(Abs(trans[2] - 0.0) > 0.0001) ||
	(Abs(trans[4] - 0.0) > 0.0001) ||
	(Abs(trans[6] - 0.0) > 0.0001) ||
	(Abs(trans[8] - 0.0) > 0.0001) ||
	(Abs(trans[9] - 0.0) > 0.0001) ||
	(Abs(trans[12] - 0.0) > 0.0001) ||
	(Abs(trans[13] - 0.0) > 0.0001) ||
	(Abs(trans[14] - 0.0) > 0.0001)) {
      axis_aligned = false;
    }
    
    // create the transform
    Transform t1, t2;
    string trans_string = "";

    for(int i=0; i<16; i++) {
      trans_string += to_string(trans[i]);
      trans_string += " ";
    }
    
    // If the nrrd had a transform as a property, and is 
    // axis_aligned data, apply that transform to the field.
    // Else if the data nrrd's transform property is different from what
    // we just calculated, then issue a warning to the user.
    string nrrd_trans_string;
    bool has_nrrd_transform = false;
    if (dataH->get_property("Transform", nrrd_trans_string) && 
	nrrd_trans_string != "Unknown" &&
	axis_aligned) {
      has_nrrd_transform = true;
      double t[16];
      int old_index=0, new_index=0;
      for(int i=0; i<16; i++) {
	new_index = nrrd_trans_string.find(" ", old_index);
	string temp = nrrd_trans_string.substr(old_index, new_index-old_index);
	old_index = new_index+1;
	string_to_double(temp, t[i]);
      }
      t2.set(t);
    } else if (dataH->get_property("Transform", nrrd_trans_string) &&
	       nrrd_trans_string != "Unknown" &&
	       nrrd_trans_string != trans_string) {
      warning("Data NRRD all ready has a transform property which is different from transform based on the Data NRRD's spaceDirections.  Using the transform based on the spaceDirection vectors.");
    }

    t1.set(trans);

    int mesh_off, pt_off;

    if (data_center == nrrdCenterCell) {
      mesh_off = 1;
      pt_off = 0;
    } else {
      mesh_off = 0;
      pt_off = 1;
    }

    topology_ = STRUCTURED;
    geometry_ = REGULAR;

    // Use these as defaults for the min and max
    // point.  They will be calculated later.  If the data
    // is not axis aligned, it won't matter what they are 
    // set to because the transform will overwrite them.
    Point minpt(0,0,0), maxpt(1,1,1);

    switch (dataH->nrrd->dim-data_off) {
    case 1:
      {
	//get data from x axis and stuff into a Scanline
	if (!has_nrrd_transform) {
	  minpt = Point (min_pt[0], 0, 0);
	  maxpt = Point ((data->axis[data_off].size - pt_off) * 
			 sp[0] + minpt.x(), 0, 0);
	}

	ScanlineMesh *mesh = scinew ScanlineMesh(data->axis[data_off].size + 
						 mesh_off, minpt, maxpt );


	// If not axis aligned, set the transform which
	// contains the origin point and the axes spaceDirection vectors
	// Otherwise, if it is axis aligned, but had a transform as a
	// property of the nrrd, apply the transform
	if (!axis_aligned)
	  mesh->set_transform(t1);
	else if (has_nrrd_transform)
	  mesh->transform(t2);

	mHandle = mesh;
      }
      break;

    case 2:
      {
	//get data from x,y axes and stuff into an Image
	if (!has_nrrd_transform) {
	  minpt = Point ( min_pt[0], min_pt[1], 0);
	  maxpt = Point( (data->axis[data_off  ].size - pt_off) * sp[0] + 
			 minpt.x(), 
			 (data->axis[data_off+1].size - pt_off) * sp[1] + 
			 minpt.y(), 0);
	}

	ImageMesh *mesh = scinew ImageMesh(data->axis[data_off  ].size + 
					   mesh_off, 
					   data->axis[data_off+1].size + 
					   mesh_off, minpt, maxpt);


	// If not axis aligned, set the transform which
	// contains the origin point and the axes spaceDirection vectors
	// Otherwise, if it is axis aligned, but had a transform as a
	// property of the nrrd, apply the transform
	if (!axis_aligned)
	  mesh->set_transform(t1);
	else if (has_nrrd_transform)
	  mesh->transform(t2);

	mHandle = mesh;
      }
      break;

    case 3:
      {
	//get data from x,y,z axes and stuff into a LatVol	
	if (!has_nrrd_transform) {
	  minpt = Point ( min_pt[0], min_pt[1], min_pt[2]);
	  maxpt = Point( (data->axis[data_off  ].size - pt_off) * sp[0] + 
			 minpt.x(), 
			 (data->axis[data_off+1].size - pt_off) * sp[1] + 
			 minpt.y(), 
			 (data->axis[data_off+2].size - pt_off) * sp[2] + 
			 minpt.z());
	}

	
	LatVolMesh *mesh = scinew LatVolMesh( 
					      data->axis[data_off  ].size + 
					      mesh_off, 
					      data->axis[data_off+1].size + 
					      mesh_off, 
					      data->axis[data_off+2].size + 
					      mesh_off, 
					      minpt, maxpt );


	// If not axis aligned, set the transform which
	// contains the origin point and the axes spaceDirection vectors
	// Otherwise, if it is axis aligned, but had a transform as a
	// property of the nrrd, apply the transform
	if (!axis_aligned)
	  mesh->set_transform(t1);	
	else if (has_nrrd_transform)
	  mesh->transform(t2);

	mHandle = mesh;
      }			   
      break;
      
    default:
      error("Cannot convert > 3 dimesional data to a SCIRun Field. If first dimension is vector/tensor data, make sure the nrrdKind is set for that axis.");
      has_error_ = true;
      return 0;
    }
    
  } else {
    // no data given
    error("Not enough information given to create a Field.");
    has_error_ = true;
    return 0;
  }
  
  
  
  // Now create field based on the mesh created above and send it
  const TypeDescription *mtd = mHandle->get_type_description();
  
  string fname = mtd->get_name();
  string::size_type pos = fname.find( "Mesh" );
  fname.replace( pos, 4, "Field" );
  
  CompileInfoHandle ci;

  if (has_data_) {
    ci = NrrdToFieldFieldAlgo::get_compile_info(mtd,
						fname,
						dataH->nrrd->type, 
						data_rank);
  } else {
    ci = NrrdToFieldFieldAlgo::get_compile_info(mtd,
						fname,
						pointsH->nrrd->type, 
						data_rank);
  }
  
  Handle<NrrdToFieldFieldAlgo> algo;
  
  if (!module_dynamic_compile(ci, algo)) return 0;
  
  if (has_data_ && (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) && build_eigens) {
    warning("Attempting to build eigen decomposition of Tensors with non symmetric tensor");
  }
  
  if (topology_ == STRUCTURED && geometry_ == IRREGULAR) {
    ofield_handle =
      algo->execute( mHandle, dataH, build_eigens, idim, jdim, kdim, permute);
    
  } else  {
    ofield_handle = algo->execute( mHandle, dataH, build_eigens);
  }
  
  return ofield_handle;  
  
}

void
 NrrdToField::tcl_command(GuiArgs& args, void* userdata)
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
NrrdToFieldMeshAlgo::get_compile_info( const string topoStr,
					      const TypeDescription *mtd,
					      const unsigned int ptype,
					      const unsigned int ctype)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string base_class_name(topoStr + "NrrdToFieldMeshAlgo");
  const string template_class_name(topoStr + "NrrdToFieldMeshAlgoT");

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
  rval->add_namespace("SCITeem");
  mtd->fill_compile_info(rval);
  return rval;
}

CompileInfoHandle
NrrdToFieldFieldAlgo::get_compile_info(const TypeDescription *mtd,
					      const string fname,
					      const unsigned int type,
					      int rank)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("NrrdToFieldFieldAlgo");
  
  string typeStr, typeName;
  
  get_nrrd_compile_type( type, typeStr, typeName );
  
  string extension;
  switch (rank)
    {
    case 6:
      extension = "Tensor";
      break;
    case 7:
      extension = "Tensor";
      break;
    case 9:
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
		       "< " + (rank==1 ? typeStr : extension) + " >" + ", " + 
		       mtd->get_name() + ", " + 
		       typeStr );
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("SCITeem");
  mtd->fill_compile_info(rval);
  return rval;
}
 
NrrdToFieldTestMeshAlgo::~NrrdToFieldTestMeshAlgo()
{
}


CompileInfoHandle
NrrdToFieldTestMeshAlgo::get_compile_info(const TypeDescription *td) 
{
  CompileInfo *rval = scinew CompileInfo(dyn_file_name(td), 
					 base_class_name(), 
					 template_class_name(), 
					 td->get_name());
  rval->add_include(get_h_file_path());
  rval->add_namespace("SCITeem");
  td->fill_compile_info(rval);
  return rval;
}

const string& 
NrrdToFieldTestMeshAlgo::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}


} // End namespace Teem



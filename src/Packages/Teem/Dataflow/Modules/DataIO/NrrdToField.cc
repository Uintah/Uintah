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

#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <iostream>

#include <Teem/Dataflow/Modules/DataIO/NrrdToField.h>

namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE NrrdToField : public Module {
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
  int         data_generation_, points_generation_;
  int         connect_generation_, origfield_generation_;
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
  // and p is the number of points per cell type.

  // build_eigens should be set to 1 if the for a field of vectors,
  // the eigendecomposition should be computed.  if data is scalar
  // or vector, pass a 0 in.

  // permute should be 1 if the data is in k x j x i format (FORTRAN)

  // For unstructured data with 4 points per cell, there is ambiguity
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
  : Module("NrrdToField", ctx, Source, "DataIO", "Teem"),
    permute_(ctx->subVar("permute")),
    build_eigens_(ctx->subVar("build-eigens")),
    quad_or_tet_(ctx->subVar("quad-or-tet")),
    struct_unstruct_(ctx->subVar("struct-unstruct")),
    data_generation_(-1), points_generation_(-1),
    connect_generation_(-1), origfield_generation_(-1),
    last_field_(0)
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

  if (!ndata_) {
    error("Unable to initialize iport 'Data'.");
    return;
  }
  if (!npoints_) {
    error("Unable to initialize iport 'Points'.");
    return;
  }
  if (!nconnect_) {
    error("Unable to initialize iport 'Connections'.");
    return;
  }
  if (!ifield_) {
    error("Unable to initialize iport 'OriginalField'.");
    return;
  }
  if (!ofield_) {
    error("Unable to initialize oport 'OutputField'.");
    return;
  }
  
  NrrdDataHandle dataH;
  NrrdDataHandle pointsH;
  NrrdDataHandle connectH;
  FieldHandle origfieldH;

  // Determine if we have data, points, connections, etc.
  if (!ndata_->get(dataH))
    dataH = 0;
  if (!npoints_->get(pointsH))
    pointsH = 0;
  if (!nconnect_->get(connectH))
    connectH = 0;
  if (!ifield_->get(origfieldH))
    origfieldH = 0;

  bool do_execute = false;
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

  if (do_execute) {
    last_field_ = create_field_from_nrrds(dataH, pointsH, connectH, origfieldH,
							build_eigens_.get(), permute_.get(), 
							quad_or_tet_.get(),
							struct_unstruct_.get());
    
  } 
  if (last_field_ != 0)
    ofield_->send(last_field_);  
}

FieldHandle 
NrrdToField::create_field_from_nrrds(NrrdDataHandle dataH, NrrdDataHandle pointsH,
				     NrrdDataHandle connectH, FieldHandle origfieldH,
				     int build_eigens, int permute, const string &quad_or_tet, 
				     const string &struct_unstruct) {
				    
  int connectivity = 0;
  MeshHandle mHandle;
  Point minpt(0,0,0), maxpt(1,1,1);
  int idim = 1, jdim = 1, kdim = 1; // initialize to 1 so it will at least go through i,j,k loops once.
  unsigned int data_rank = 0;
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


  ///////////// HAS ORIGINATING FIELD /////////////

  data_rank = 1;
  if (has_origfield_) {
    // Attempt to use the field provided
    
    if (has_data_) {
      // Based on the gui checkbox indicating whether or not
      // the first dimension is vector/tensor data, alter
      // the dim accordingly.
      int non_scalar_data = 0;
      if (nrrdKindSize( dataH->nrrd->axis[0].kind ) > 1) {
	non_scalar_data = 1;
      }
      
      if (non_scalar_data && has_data_) {
	if (dataH->nrrd->axis[0].size == 3) {
	  data_rank = 3;
	} else if (dataH->nrrd->axis[0].size >= 6) {
	  data_rank = dataH->nrrd->axis[0].size;
	} else {
	  return 0;
	}
      }
    } else {
      dataH = 0;
    }

    mHandle = origfieldH->mesh();

    // verify the data will "fit" into the mesh passed in
    if (has_data_) {
      const TypeDescription *td = origfieldH->get_type_description();
      CompileInfoHandle ci = NrrdToFieldTestMeshAlgo::get_compile_info(td);
      Handle<NrrdToFieldTestMeshAlgo> algo;
      if ((module_dynamic_compile(ci, algo)) && 
	  (algo->execute(origfieldH, dataH, ofield_handle, data_rank))) 
	{
	  remark("Creating a Field from original mesh in input nrrd");
	  
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
	}
    } else {
      dataH = 0;
      
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
      
      ofield_handle = algo->execute( mHandle, dataH, build_eigens);
      
      return ofield_handle;
    }
    cerr << "Tried to use field but failed\n";
  }
  





  /////////////////// DETERMINE MESH BASED ON INPUTS AND GUI INFO ///////////////////
  int non_scalar_data = 0;
  if (has_data_ && nrrdKindSize( dataH->nrrd->axis[0].kind ) > 1) {
    non_scalar_data = 1;
  }
  
  if (non_scalar_data && has_data_) {
    if (dataH->nrrd->axis[0].size == 3) {
      data_rank = 3;
    } else if (dataH->nrrd->axis[0].size >= 6) {
      data_rank = dataH->nrrd->axis[0].size;
    } else {
      return 0;
    }
  }

  // No field given, does it have connections?
  if (has_connect_) {
    if (has_points_) {
      Nrrd* connect = connectH->nrrd;
      // Look at connections nrrd's 2nd dimension
      // account for vector/scalar data
      if (connect->dim != 2) {
	error("Connections Nrrd must be two dimensional (number of points in each connection by the number of connections)");
	return 0;
      }
      
      // check if connect array is p x n or n x p
      int which = 0; // which index contains p
      if (connect->axis[1].size <= 8 && connect->axis[0].size <= 8) {
	warning("Connections nrrd might be in the wrong order.  Assuming p x n where p is the number of points in each connection and n is the number of connections.");
      }
      if (connect->axis[1].size <= 8) {
	// p x n
	which = 1;
      } else if (connect->axis[0].size <= 8) {
	// p x n
	which = 0;
      } else { 
	error("Connections nrrd must have one axis with a size less than or equal to 8.");
	return 0;
      }
      
      int pts = connect->axis[which].size;
      
      switch (pts) {
      case 2:
	// 2 -> curve
	mHandle = scinew CurveMesh();
	connectivity = 2;
	topology_ = STRUCTURED;
	geometry_ = IRREGULAR;
	break;
      case 3:
	// 3 -> tri
	mHandle = scinew TriSurfMesh();
	connectivity = 3;
	topology_ = UNSTRUCTURED;
	break;
      case 4: {
	// 4 -> quad/tet (ask which if this case)
	string cell_type = "Auto";
	if (quad_or_tet == "Auto") {
	  // determine if tet/quad based on properties
	  if(pointsH->get_property( "Cell Type", property )) {
	    if( property.find( "Tet" ) != string::npos )
	      cell_type = "Tet";
	    else if( property.find( "Quad" ) != string::npos )
	      cell_type = "Quad";
	  } else if (connectH->get_property( "Cell Type",property )) {
	    if( property.find( "Tet" ) != string::npos )
	      cell_type = "Tet";
	    else if( property.find( "Quad" ) != string::npos )
	      cell_type = "Quad";
	  } else if (has_data_ && dataH->get_property( "Cell Type", property )) {
	    if( property.find( "Tet" ) != string::npos )
	      cell_type = "Tet";
	    else if( property.find( "Quad" ) != string::npos )
	      cell_type = "Quad";
	  } else {
	    warning("Auto detection of Cell Type using properties failed.  Using Tet for ambiguious case of 4 points per connection.");
	    cell_type = "Tet";
	  }
	  
	  if (cell_type == "Tet") {
	    mHandle = scinew TetVolMesh();
	    connectivity = 4;	    
	  } else if (cell_type != "Quad") {
	    mHandle = scinew QuadSurfMesh();
	    connectivity = 4;
	  } else {
	    error("Connections Nrrd indicates 4 points per connection. Please indicate whether a Tet or Quad in UI.");
	    return 0;
	  }
	  topology_ = UNSTRUCTURED;
	}
	else if (quad_or_tet == "Quad") {
	  mHandle = scinew QuadSurfMesh();
	  connectivity = 4;
	  topology_ = UNSTRUCTURED;
	} else {
	  mHandle = scinew TetVolMesh();
	  connectivity = 4;
	  topology_ = UNSTRUCTURED;
	}
      }
	break;
      case 6:
	// 6 -> prism
	mHandle = scinew PrismVolMesh();
	connectivity = 6;
	topology_ = UNSTRUCTURED;
	break;
      case 8:
	// 8 -> hex
	mHandle = scinew HexVolMesh();
	connectivity = 8;
	topology_ = UNSTRUCTURED;
	break;
      default:
	error("Connections Nrrd must contain 2, 3, 4, 6, or 8 points per connection.");
	return 0;
      }
      
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
      
    }
  } else {
    // no connections, does it have data?
    if (has_data_) {
      // does it have a point set
      if (has_points_) {
	int offset = 0;
	// data and points must have the same number of dimensions and the all but the first
	// must be the same size
	int d = 0, p = 0;
	for (d = dataH->nrrd->dim-1, p = pointsH->nrrd->dim-1; p > 0; d--, p--) {
	  if (dataH->nrrd->axis[d].size != pointsH->nrrd->axis[p].size) {
	    error("Data and Points must have the same dimension size for all but the first axis.");
	    return 0;
	  }
	}
	
	if (nrrdKindSize(dataH->nrrd->axis[0].kind) > 1) {
	  offset = 1;
	}
	Nrrd* data = dataH->nrrd;
	int dim = data->dim - offset;
	if (dim == 1) {
	  // data 1D ask if point cloud or structcurvemesh
	  if (struct_unstruct == "PointCloud") {
	    mHandle = scinew PointCloudMesh();
	    connectivity = 0;
	    topology_ = UNSTRUCTURED;
	  } else if (struct_unstruct == "StructCurve") {
	    topology_ = STRUCTURED;
	    geometry_ = IRREGULAR;
	    if (offset) {
	      mHandle = scinew StructCurveMesh( data->axis[1].size );
	      idim = data->axis[1].size;
	    }
	    else {
	      mHandle = scinew StructCurveMesh( data->axis[0].size );
	      idim = data->axis[0].size;
	    }
	  } else {
	    // search properties if auto to figure out
	    if( pointsH->get_property( "Topology", property ) ) {
	      if( property.find( "Unstructured" ) != string::npos ) {
		mHandle = scinew PointCloudMesh();
		connectivity = 0;
		topology_ = UNSTRUCTURED;	      
	      } else {
		topology_ = STRUCTURED;
		geometry_ = IRREGULAR;
		if (offset) {
		  mHandle = scinew StructCurveMesh( data->axis[1].size );
		  idim = data->axis[1].size;
		}
		else {
		  mHandle = scinew StructCurveMesh( data->axis[0].size );
		  idim = data->axis[0].size;
		}
	      }
	    } else if (dataH->get_property( "Topology", property )) {
	      if( property.find( "Unstructured" ) != string::npos ) {
		mHandle = scinew PointCloudMesh();
		connectivity = 0;
		topology_ = UNSTRUCTURED;	      
	      } else {
		topology_ = STRUCTURED;
		geometry_ = IRREGULAR;
		if (offset) {
		  mHandle = scinew StructCurveMesh( data->axis[1].size );
		  idim = data->axis[1].size;
		}
		else {
		  mHandle = scinew StructCurveMesh( data->axis[0].size );
		  idim = data->axis[0].size;
		}
	      }	      
	    } else if (has_connect_ && connectH->get_property( "Topology", property)) {
	      if( property.find( "Unstructured" ) != string::npos ) {
		mHandle = scinew PointCloudMesh();
		connectivity = 0;
		topology_ = UNSTRUCTURED;	      
	      } else {
		topology_ = STRUCTURED;
		geometry_ = IRREGULAR;
		if (offset) {
		  mHandle = scinew StructCurveMesh( data->axis[1].size );
		  idim = data->axis[1].size;
		}
		else {
		  mHandle = scinew StructCurveMesh( data->axis[0].size );
		  idim = data->axis[0].size;
		}
	      }	      
	    } else {
	      warning("Unable to determine if creating Point Cloud or Struct Curve. Defaulting to Point Cloud");
	      mHandle = scinew PointCloudMesh();
	      connectivity = 0;
	    }
	  } 
	} else if (dim == 2) {
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  // data 2D -> structquad
	  if (offset) {
	    mHandle = scinew StructQuadSurfMesh(data->axis[1].size, data->axis[2].size);
	    idim = data->axis[1].size;
	    jdim = data->axis[2].size;
	    //kdim = 1;
	  } else {
	    mHandle = scinew StructQuadSurfMesh(data->axis[0].size, data->axis[1].size);
	    idim = data->axis[0].size;
	    jdim = data->axis[1].size;
	    //kdim = 1;
	  }
	} else if (dim == 3) {
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  // data 3D -> structhexvol
	  if (offset) {
	    mHandle = scinew StructHexVolMesh(data->axis[1].size, data->axis[2].size,
					      data->axis[3].size);
	    idim = data->axis[1].size;
	    jdim = data->axis[2].size;
	    kdim = data->axis[3].size;
	  } else {
	    mHandle = scinew StructHexVolMesh(data->axis[0].size, data->axis[1].size,
					      data->axis[2].size);
	    idim = data->axis[0].size;
	    jdim = data->axis[1].size;
	    kdim = data->axis[2].size;
	  }
	} else {
	  error("Incorrect dimensions for Data Nrrd");
	  return 0;
	}

	// create mesh
	const TypeDescription *mtd = mHandle->get_type_description();
	
	remark( "Creating a structured " + mtd->get_name() );
	
	CompileInfoHandle ci_mesh =
	  NrrdToFieldMeshAlgo::get_compile_info( "Structured",
							mtd,
							pointsH->nrrd->type,
							pointsH->nrrd->type);
	
	Handle<StructuredNrrdToFieldMeshAlgo> algo_mesh;
	
	if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return 0;

	int data_size = 1;
	if (has_data_ && nrrdKindSize(dataH->nrrd->axis[0].kind) > 1)
	  data_size = dataH->nrrd->axis[0].size;
	
	algo_mesh->execute(mHandle, pointsH, idim, jdim, kdim );
      } else {
	int non_scalar_data = 0;
	if (has_data_ && nrrdKindSize(dataH->nrrd->axis[0].kind) > 1) {
	  non_scalar_data = 1;
	}
	int dim = dataH->nrrd->dim;
	int offset = 0;
	if (non_scalar_data) {
	  offset = 1;
	  remark("First dimension is vector/tensor data");
	  --dim;
	}
	
	Nrrd* data = dataH->nrrd;
	bool has_min_pt = true;
	vector<double> sp;
	int data_center = nrrdCenterUnknown;
	int mesh_off = 0, pt_off = 0;
	
	if (has_data_) {
	  for (int i=offset; i<dataH->nrrd->dim; i++) {
	    if (!(AIR_EXISTS(data->axis[i].min) && AIR_EXISTS(data->axis[i].max))) {
	      has_min_pt = false;
	      nrrdAxisInfoMinMaxSet(data, i, nrrdCenterNode);
	    }
	    if (AIR_EXISTS(data->axis[i].spacing))
	      sp.push_back(data->axis[i].spacing);
	    else
	      sp.push_back(1.0);
	    if (data->axis[i].center != nrrdCenterUnknown) 
	      data_center = data->axis[i].center;
	  }
	} else {
	  has_min_pt = false;
	}

	if (data_center == nrrdCenterCell) 
	  mesh_off = 1;
	else 
	  pt_off = 1;
	switch (dim) {
	case 1:
	  {
	    //get data from x axis and stuff into a Scanline
	    topology_ = STRUCTURED;
	    geometry_ = REGULAR;
	    if (offset && data->dim == 2) {
	      if (has_min_pt)
		minpt = Point (data->axis[1].min, 0, 0);
	      maxpt = Point ((data->axis[1].size - pt_off) * sp[0] + minpt.x(), 0, 0);
	      mHandle = scinew ScanlineMesh( data->axis[1].size + mesh_off, minpt, maxpt );
	    } else {
	      if (has_min_pt)
		minpt = Point (data->axis[0].min, 0, 0);
	      maxpt = Point ((data->axis[0].size - pt_off) * sp[0] + minpt.x(), 0, 0);
	      mHandle = scinew ScanlineMesh( data->axis[0].size + mesh_off, minpt, maxpt );
	    }
	  }
	  break;
	case 2:
	  {
	    //get data from x,y axes and stuff into an Image
	    topology_ = STRUCTURED;
	    geometry_ = REGULAR;
	    if (offset && data->dim == 3) {
	      if (has_min_pt)
		minpt = Point ( data->axis[1].min, data->axis[2].min, 0);
	      maxpt = Point( (data->axis[1].size - pt_off) * sp[0] + minpt.x(), 
			     (data->axis[2].size - pt_off) * sp[1] + minpt.y(), 0);
	      mHandle = scinew ImageMesh( data->axis[1].size + mesh_off, 
					  data->axis[2].size + mesh_off, 
					  minpt, maxpt);
	    } else {
	      if (has_min_pt)
		minpt = Point ( data->axis[0].min, data->axis[1].min, 0);
	      maxpt = Point( (data->axis[0].size - pt_off) * sp[0] + minpt.x(), 
			     (data->axis[1].size - pt_off) * sp[1] + minpt.y(), 0);	      
	      mHandle = scinew ImageMesh( data->axis[0].size + mesh_off, 
					  data->axis[1].size + mesh_off, 
					  minpt, maxpt);
	    }
	  }
	  break;
	case 3:
	  {
	    //get data from x,y,z axes and stuff into a LatVol
	    topology_ = STRUCTURED;
	    geometry_ = REGULAR;
	    if (offset && data->dim == 4) {
	      if (has_min_pt) 
		minpt = Point ( data->axis[1].min, data->axis[2].min, data->axis[3].min);
	      maxpt = Point( (data->axis[1].size - pt_off) * sp[0] + minpt.x(), 
			     (data->axis[2].size - pt_off) * sp[1] + minpt.y(), 
			     (data->axis[3].size - pt_off) * sp[2] + minpt.z());
	      mHandle = scinew LatVolMesh( data->axis[1].size + mesh_off, 
					   data->axis[2].size + mesh_off, 
					   data->axis[3].size + mesh_off, 
					   minpt, maxpt );
	    } else {
	      if (has_min_pt) 
		minpt = Point ( data->axis[0].min, data->axis[1].min, data->axis[2].min);
	      maxpt = Point( (data->axis[0].size - pt_off) * sp[0] + minpt.x(), 
			     (data->axis[1].size - pt_off) * sp[1] + minpt.y(), 
			     (data->axis[2].size - pt_off) * sp[2] + minpt.z());
	      mHandle = scinew LatVolMesh( data->axis[0].size + mesh_off, 
					   data->axis[1].size + mesh_off, 
					   data->axis[2].size + mesh_off, 
					   minpt, maxpt );
	    }
	  }
	  break;
	default:
	  error("Cannot convert > 3 dimesional data to a SCIRun Field. If first dimension is vector/tensor data, make sure the nrrdKind is set for that axis.");
	  return 0;
	}
      }
    } else {
      // does it have points?
      if (has_points_) {
	// ask if point cloud, structcurve, structquad, structhex
	// in the case of structquad, need ni and nj  and possible nk from points axes sizes/
	// these should corresond to the sizes of the corresponsing points axes
	
	Nrrd* points = pointsH->nrrd;

	if (pointsH->nrrd->dim == 1) {
	  string which = struct_unstruct;
	  if (which == "PointCloud") {
	    mHandle = scinew PointCloudMesh();
	    topology_ = UNSTRUCTURED;
	  } else if (which == "StructCurve") {
	    topology_ = STRUCTURED;
	    geometry_ = IRREGULAR;
	    
	    mHandle = scinew StructCurveMesh( points->axis[1].size );
	    idim = points->axis[1].size;
	  } else {
	    // Try to figure out based on properties
	    if (pointsH->get_property( "Topology" , property)) {
	      if( property.find( "Unstructured" ) != string::npos ) {
		topology_ = UNSTRUCTURED;
		mHandle = scinew PointCloudMesh();
	      } else {
		if ( property.find( "Structured" ) != string::npos ) {
		  topology_ = STRUCTURED;
		  geometry_ = IRREGULAR;
		  if (pointsH->get_property( "Cell Type", property)) { 
		    if ( property.find( "Curve") != string::npos ) {
		      // Struct Curve
		      mHandle = scinew StructCurveMesh( points->axis[1].size );
		      idim = points->axis[1].size;
		    } 
		  } else {
		    // Struct Curve Mesh
		    mHandle = scinew StructCurveMesh( points->axis[1].size );
		    idim = points->axis[1].size;
		  }
		}
	      }
	    }
	  }
	} else if (pointsH->nrrd->dim == 2) {
	  topology_ = STRUCTURED;
 	  geometry_ = IRREGULAR;
	  mHandle = scinew StructQuadSurfMesh( points->axis[1].size, points->axis[2].size );
	  idim = points->axis[1].size;
	  jdim = points->axis[2].size;
	} else if (pointsH->nrrd->dim == 3) {
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  mHandle = scinew StructHexVolMesh( points->axis[1].size, points->axis[2].size,
					     points->axis[3].size );
	  idim = points->axis[1].size;
	  jdim = points->axis[2].size;
	  kdim = points->axis[3].size;
	} else {
	  error("Points nrrd must have 1, 2, or 3 dimensions.");
	  return 0;
	}
      } else {
	// no data given
	error("Not enough information given to create a Field.");
	return 0;
      }
    }
  }
  
  // Now create field based on the mesh created above and send it
  const TypeDescription *mtd = mHandle->get_type_description();
  
  string fname = mtd->get_name();
  string::size_type pos = fname.find( "Mesh" );
  fname.replace( pos, 4, "Field" );
  
  //data_rank = 1;
  
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
    dataH = 0;
  }
  
  Handle<NrrdToFieldFieldAlgo> algo;
  
  if (!module_dynamic_compile(ci, algo)) return 0;
  
  if (has_data_ && (nrrdKindSize( dataH->nrrd->axis[0].kind) == 9) && build_eigens) {
    warning("Attempting to build eigendecomposition of Tensors with non symmetric tensor");
  }
  
  if (topology_ == STRUCTURED && geometry_ == IRREGULAR) {
    ofield_handle = algo->execute( mHandle, dataH, build_eigens, idim, jdim, kdim, permute);
    
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



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

//#include <Teem/Dataflow/Modules/DataIO/ConvertToField.h>
//#include <Teem/Dataflow/Modules/DataIO/NrrdFieldConverter.h>
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
  GuiInt      non_scalar_data_;
  GuiInt      build_eigens_;
  GuiString   quad_or_tet_;
  GuiString   struct_unstruct_;
  GuiInt      ni_, nj_, nk_;
  int         has_data_, has_points_, has_connect_, has_origfield_;
  int         data_generation_, points_generation_;
  int         connect_generation_, origfield_generation_;

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
  // nrrd should be a 2D array of nxp where n is the number of connections
  // and p is the number of points per cell type.
  // permute indicates whether the data should be permuted because it
  // is stored as KxJxI.
  // For unstructured data with 4 points per cell, there is ambiguity
  // whether or not it is a tet or quad.  Set to "Auto" to try to 
  // determine based on Nrrd properties or "Tet" or "Quad".  If it
  // in unclear and "Auto" has been selected, it will default to "Tet".
  // In the case of only points, it must be indicated whether it is
  // a Point Cloud, StructCurve, StructQuad, or StructHex.  For the
  // Quad and Hex, an ni, nj, and nk are needed.  Pass the string 
  // "Auto" to determine based on Nrrd properties, "StructCurve",
  // "StructQuad" or "StructHex".  If "Auto" is selected and it cannot
  // be determined, a PointCloud will be created.
  // In the case of having data and points with the points being an nx1
  // nrrd, it is unclear if it should be a PointCloud or StructCurve.  
  // Pass the string "Auto" to determine from the Nrrd Properties, or
  // "PointCloud" or "StructCurve".  If "Auto" is selected and it
  // cannot be determined, then it will default to a Point Cloud.
  // ni, nj, and nk represent the size if it is a StructQuad or StructHex.
  FieldHandle create_field_from_nrrds(NrrdDataHandle dataH,
				      NrrdDataHandle pointsH,
				      NrrdDataHandle connectH,
				      FieldHandle    origfieldH,
				      int permute,
				      string quad_or_tet,
				      string struct_unstruct,
				      int ni, int nj, int nk);
  
};


DECLARE_MAKER(NrrdToField)
NrrdToField::NrrdToField(GuiContext* ctx)
  : Module("NrrdToField", ctx, Source, "DataIO", "Teem"),
    permute_(ctx->subVar("permute")),
    non_scalar_data_(ctx->subVar("non-scalar-data")),
    build_eigens_(ctx->subVar("build-eigens")),
    quad_or_tet_(ctx->subVar("quad-or-tet")),
    struct_unstruct_(ctx->subVar("struct-unstruct")),
    ni_(ctx->subVar("ni")),
    nj_(ctx->subVar("nj")),
    nk_(ctx->subVar("nk")),
    has_data_(0), has_points_(0),
    has_connect_(0), has_origfield_(0),
    data_generation_(-1), points_generation_(-1),
    connect_generation_(-1), origfield_generation_(-1)
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
  
  cerr << "FIX ME : NEED TO CHECK GENERATIONS BEFORE EXECUTING\n";

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

  FieldHandle ofield_handle = create_field_from_nrrds(dataH, pointsH, connectH, origfieldH,
						      permute_.get(), quad_or_tet_.get(),
						      struct_unstruct_.get(), ni_.get(),
						      nj_.get(), nk_.get());

  ofield_->send(ofield_handle);  
}

FieldHandle 
NrrdToField::create_field_from_nrrds(NrrdDataHandle dataH, NrrdDataHandle pointsH,
				    NrrdDataHandle connectH, FieldHandle origfieldH,
				    int permute, string quad_or_tet, string struct_unstruct,
				      int ni, int nj, int nk) {
				    
  int connectivity = 0;
  MeshHandle mHandle;
  Point minpt(0,0,0), maxpt(1,1,1);
  int idim = 0, jdim = 0, kdim = 0;
  unsigned int data_rank = 0;
  string property;
  unsigned int topology_ = UNKNOWN;
  unsigned int geometry_ = UNKNOWN;
  FieldHandle ofield_handle = 0;
  

  // Determine if we have data, points, connections, etc.
  has_data_ = has_points_ = has_connect_ = has_origfield_ = false;

  if (dataH != 0)
    has_data_ = true;
  if (pointsH != 0)
    has_points_ = true;
  if (connectH != 0)
    has_connect_ = true;
  if (has_origfield_ != 0)
    has_origfield_ = true;


  ///////////// HAS ORIGINATING FIELD /////////////

  data_rank = 1;
  if (has_origfield_) {
    // Attempt to use the field provided
    cerr << "Use Field provided\n";
    
    if (has_data_) {
      // Based on the gui checkbox indicating whether or not
      // the first dimension is vector/tensor data, alter
      // the dim accordingly.
      int non_scalar_data = non_scalar_data_.get();
      
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
    
    
    cerr << "Creating FIELD2\n";
    // Now create field based on the mesh created above and send it
    const TypeDescription *mtd = mHandle->get_type_description();
    
    string fname = mtd->get_name();
    string::size_type pos = fname.find( "Mesh" );
    fname.replace( pos, 4, "Field" );
    
    CompileInfoHandle ci;
    
    if (has_data_) {
      ci = NrrdFieldConverterFieldAlgo::get_compile_info(mtd,
							 fname,
							 dataH->nrrd->type, 
							 data_rank);
    } else {
      ci = NrrdFieldConverterFieldAlgo::get_compile_info(mtd,
							 fname,
							 pointsH->nrrd->type, 
							 data_rank);
    }
    
    Handle<NrrdFieldConverterFieldAlgo> algo;
    
    if (!module_dynamic_compile(ci, algo)) return 0;
    
    if (!has_data_) 
      dataH = 0;
    
    //if( topology_ & STRUCTURED ) {
    if (idim != 0) { // has been changed 
      ofield_handle = algo->execute( mHandle, dataH, idim, jdim, kdim, permute);
      
    } else  {
      ofield_handle = algo->execute( mHandle, dataH );
    }
    
    return ofield_handle;
  }






  /////////////////// DETERMINE MESH BASED ON INPUTS AND GUI INFO ///////////////////
  int non_scalar_data = non_scalar_data_.get();
  
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
      cerr << "Creating curve, tri, quad/tet, or hex\n";
      Nrrd* connect = connectH->nrrd;
      // Look at connections nrrd's 2nd dimension
      // account for vector/scalar data
      if (connect->dim != 2) {
	error("Connections Nrrd must be two dimensional (number of connections by number of points in each connection)");
	return 0;
      }
      
      int pts = connect->axis[1].size;
      
      switch (pts) {
      case 2:
	// 2 -> curve
	cerr << "\tCurveMesh\n";
	mHandle = scinew CurveMesh();
	connectivity = 2;
	topology_ = STRUCTURED;
	geometry_ = IRREGULAR;
	break;
      case 3:
	// 3 -> tri
	cerr << "\tTriSurfMesh\n";
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
	  cerr << "\tQuadSurfMesh\n";
	  mHandle = scinew QuadSurfMesh();
	  connectivity = 4;
	  topology_ = UNSTRUCTURED;
	} else {
	  cerr << "\tTetVolMesh\n";
	  mHandle = scinew TetVolMesh();
	  connectivity = 4;
	  topology_ = UNSTRUCTURED;
	}
      }
	break;
      case 6:
	// 6 -> prism
	cerr << "\tPrismVolMesh\n";
	mHandle = scinew PrismVolMesh();
	connectivity = 6;
	topology_ = UNSTRUCTURED;
	break;
      case 8:
	// 8 -> hex
	cerr << "\tHexVolMesh\n";
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
	NrrdFieldConverterMeshAlgo::get_compile_info("Unstructured",
						     mtd,
						     pointsH->nrrd->type,
						     connectH->nrrd->type);
      
      Handle<UnstructuredNrrdFieldConverterMeshAlgo> algo_mesh;
      
      if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return 0;

      //int data_size = 1;
      //if (non_scalar_data_.get() && has_data_)
      //data_size = dataH->nrrd->axis[0].size;

      //algo_mesh->execute(mHandle, pointsH, connectH, connectivity, data_size);
      algo_mesh->execute(mHandle, pointsH, connectH, connectivity);

    }
  } else {
    // no connections, does it have data?
    if (has_data_) {
      // does it have a point set
      if (has_points_) {
	cerr << "Creating point cloud, structcurve, structquad, or structhex\n";
	int offset = 0;
	if (non_scalar_data_.get()) {
	  cerr << "Vector/Tensor data\n";
	  offset = 1;
	}
	Nrrd* data = dataH->nrrd;
	int dim = data->dim - offset;
	if (dim == 1) {
	  // data 1D ask if point cloud or structcurvemesh
	  if (struct_unstruct == "PointCloud") {
	    cerr << "\tPoint Cloud\n";
	    mHandle = scinew PointCloudMesh();
	    connectivity = 0;
	    topology_ = UNSTRUCTURED;
	  } else if (struct_unstruct == "StructCurve") {
	    cerr << "\tStructCurveMesh\n";
	    topology_ = STRUCTURED;
	    geometry_ = IRREGULAR;
	    if (offset)
	      mHandle = scinew StructCurveMesh( data->axis[1].size );
	    else
	      mHandle = scinew StructCurveMesh( data->axis[0].size );
	  } else {
	    // search properties if auto to figure out
	    if( pointsH->get_property( "Topology", property ) ) {
	      if( property.find( "Unstructured" ) != string::npos ) {
		cerr << "\tPoint Cloud\n";
		mHandle = scinew PointCloudMesh();
		connectivity = 0;
		topology_ = UNSTRUCTURED;	      
	      } else {
		cerr << "\tStructCurveMesh\n";
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
		cerr << "\tPoint Cloud\n";
		mHandle = scinew PointCloudMesh();
		connectivity = 0;
		topology_ = UNSTRUCTURED;	      
	      } else {
		cerr << "\tStructCurveMesh\n";
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
		cerr << "\tPoint Cloud\n";
		mHandle = scinew PointCloudMesh();
		connectivity = 0;
		topology_ = UNSTRUCTURED;	      
	      } else {
		cerr << "\tStructCurveMesh\n";
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
	  cerr << "\tStructQuadSurf\n";
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  // data 2D -> structquad
	  if (offset) {
	    cerr << "FIX ME : INCORPORATE THE ni, nj, nk FROM UI\n";
	    mHandle = scinew StructQuadSurfMesh(data->axis[1].size, data->axis[2].size);
	    idim = data->axis[1].size;
	    jdim = data->axis[2].size;
	  }
	  else {
	    mHandle = scinew StructQuadSurfMesh(data->axis[0].size, data->axis[1].size);
	    idim = data->axis[0].size;
	    jdim = data->axis[1].size;
	  }
	} else if (dim == 3) {
	  cerr << "\tStructHexVol\n";
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  // data 3D -> structhexvol
	  if (offset) {
	    mHandle = scinew StructHexVolMesh(data->axis[1].size, data->axis[2].size, data->axis[3].size);
	    idim = data->axis[1].size;
	    jdim = data->axis[2].size;
	    kdim = data->axis[3].size;
	  }
	  else {
	    mHandle = scinew StructHexVolMesh(data->axis[0].size, data->axis[1].size, data->axis[2].size);
	    idim = data->axis[0].size;
	    jdim = data->axis[1].size;
	    kdim = data->axis[2].size;
	  }
	} else {
	  cerr << "Incorrect dimensions for Data Nrrd\n";
	  return 0;
	}

	// create mesh
	const TypeDescription *mtd = mHandle->get_type_description();
	
	remark( "Creating a structured " + mtd->get_name() );
	
	CompileInfoHandle ci_mesh =
	  NrrdFieldConverterMeshAlgo::get_compile_info( "Structured",
							mtd,
							pointsH->nrrd->type,
							pointsH->nrrd->type);
	
	Handle<StructuredNrrdFieldConverterMeshAlgo> algo_mesh;
	
	if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return 0;

	int data_size = 1;
	if (non_scalar_data_.get() && has_data_)
	  data_size = dataH->nrrd->axis[0].size;
	
	algo_mesh->execute(mHandle, pointsH, idim, jdim, kdim );
      } else {
	cerr << "Creating scanline, image, or latvol\n";
	int non_scalar_data = non_scalar_data_.get();
	int dim = dataH->nrrd->dim;
	int offset = 0;
	if (non_scalar_data) {
	  offset = 1;
	  remark("First dimension is vector/tensor data");
	  --dim;
	}
	
	Nrrd* data = dataH->nrrd;

	switch (dim) {
	case 1:
	  {
	    //get data from x axis and stuff into a Scanline
	    cerr << "\tCreating a ScanlineField from input nrrd";
	    topology_ = STRUCTURED;
	    geometry_ = REGULAR;
	    if (offset && data->dim == 2) {
	      maxpt = Point (data->axis[1].size, 0, 0);
	      mHandle = scinew ScanlineMesh( data->axis[1].size, minpt, maxpt );
	    } else {
	      maxpt = Point (data->axis[0].size, 0, 0);
	      mHandle = scinew ScanlineMesh( data->axis[0].size, minpt, maxpt );
	    }
	  }
	  break;
	case 2:
	  {
	    //get data from x,y axes and stuff into an Image
	    cerr << "\tCreating a ImageField from input nrrd";
	    topology_ = STRUCTURED;
	    geometry_ = REGULAR;
	    if (offset && data->dim == 3) {
	      maxpt = Point( data->axis[1].size, data->axis[2].size, 0);
	      mHandle = scinew ImageMesh( data->axis[1].size, data->axis[2].size, minpt, maxpt);
	    } else {
	      maxpt = Point( data->axis[0].size, data->axis[1].size, 0);
	      mHandle = scinew ImageMesh( data->axis[0].size, data->axis[1].size, minpt, maxpt);
	    }
	  }
	  break;
	case 3:
	  {
	    //get data from x,y,z axes and stuff into a LatVol
	    cerr << "\tCreating a LatVolField from input nrrd";
	    topology_ = STRUCTURED;
	    geometry_ = REGULAR;
	    if (offset && data->dim == 4) {
	      maxpt = Point( data->axis[1].size, data->axis[2].size, data->axis[3].size ); 	
	      mHandle = scinew LatVolMesh( data->axis[1].size, data->axis[2].size, data->axis[3].size, minpt, maxpt );
	    } else {
	      maxpt = Point( data->axis[0].size, data->axis[1].size, data->axis[2].size ); 	
	      mHandle = scinew LatVolMesh( data->axis[0].size, data->axis[1].size, data->axis[2].size, minpt, maxpt );
	    }
	  }
	  break;
	default:
	  error("Cannot convert > 3 dimesional data to a SCIRun Field. If first dimension is vector or tensor data, please indicate on NrrdToField UI.");
	  return 0;
	}

// 	cerr << "Creating the mesh...\n";
// 	// create the mesh
// 	const TypeDescription *mtd = mHandle->get_type_description();
// 	int data_size = 1; // scalar
// 	if (offset) 
// 	  data_size = data->axis[0].size;

	
// 	remark( "Creating a structured " + mtd->get_name() );
	
// 	CompileInfoHandle ci_mesh =
// 	  NrrdFieldConverterMeshAlgo::get_compile_info( "Regular",
// 							mtd,
// 							data->type,
// 							data->type);
// 	cerr << "Done with compileinfohandle\n";
	
// 	Handle<RegularNrrdFieldConverterMeshAlgo> algo_mesh;
	
// 	if( !module_dynamic_compile(ci_mesh, algo_mesh) ) return;
// 	cerr << "Done dynamically compiling\n";
	
	// algo_mesh->execute(mHandle, dataH, data_size);
	
      }
    } else {
      // does it have points?
      if (has_points_) {
	cerr << "Creating point cloud, structcurve, structquad, structhex\n";
	// ask if point cloud, structcurve, structquad, structhex
	// in the case of structquad, need ni and nj
	// in the case of structhex, need ni, nj, and nk
	
	Nrrd* points = pointsH->nrrd;
	int offset = 0;
	if (non_scalar_data_.get()) {
	  cerr << "Vector/Tensor data\n";
	  offset = 1;
	}
	string which = struct_unstruct;
	if (which == "PointCloud") {
	  cerr << "\tPoint Cloud\n";
	  mHandle = scinew PointCloudMesh();
	    topology_ = UNSTRUCTURED;
	} else if (which == "StructCurve") {
	  cerr << "\tStruct Curve\n";
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  if (offset) {
	    mHandle = scinew StructCurveMesh( points->axis[1].size );
	  } else {
	    mHandle = scinew StructCurveMesh( points->axis[0].size );
	  }
	} else if (which == "StructQuad") {
	  cerr << "\tStruct Quad\n";
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  if (offset) {
	    mHandle = scinew StructQuadSurfMesh( points->axis[1].size, points->axis[2].size );
	  } else {
	    mHandle = scinew StructQuadSurfMesh( points->axis[0].size, points->axis[1].size );
	  }
	} else if (which == "StructHex") {
	  cerr << "\tStruct Hex\n";
	  topology_ = STRUCTURED;
	  geometry_ = IRREGULAR;
	  if (offset) {
	    mHandle = scinew StructHexVolMesh( points->axis[1].size, points->axis[2].size, points->axis[3].size );
	  } else {
	    mHandle = scinew StructHexVolMesh( points->axis[0].size, points->axis[1].size, points->axis[2].size );
	  }
	} else {
	  // Try to figure out based on properties
	  if (pointsH->get_property( "Topology" , property)) {
	    if( property.find( "Unstructured" ) != string::npos ) {
	      cerr << "\tPoint Cloud\n";
	      topology_ = UNSTRUCTURED;
	      mHandle = scinew PointCloudMesh();
	    } else {
	      cerr << "FIX ME : NOT SURE THIS IS CORRECT\n";
	    }
	  }
	}
      } else {
	// no data given
	error("Not enough information given to create a Field.");
	return 0;
      }
    }
  }
  
  cerr << "Creating FIELD\n";
  // Now create field based on the mesh created above and send it
  const TypeDescription *mtd = mHandle->get_type_description();
  
  string fname = mtd->get_name();
  string::size_type pos = fname.find( "Mesh" );
  fname.replace( pos, 4, "Field" );
  
  //data_rank = 1;
  
  CompileInfoHandle ci;
  
  // if (non_scalar_data_.get() && has_data_) {
  //if (dataH->nrrd->axis[0].size == 3) {
  //data_rank = 3;
  //} else if (dataH->nrrd->axis[0].size >= 6) {
  //data_rank = dataH->nrrd->axis[1].size;
  //} else {
  //return;
  //}
  //}
  cerr << "Data rank: " << data_rank << "\n";
  if (has_data_) {
    ci = NrrdFieldConverterFieldAlgo::get_compile_info(mtd,
						       fname,
						       dataH->nrrd->type, 
						       data_rank);
  } else {
    ci = NrrdFieldConverterFieldAlgo::get_compile_info(mtd,
						       fname,
						       pointsH->nrrd->type, 
						       data_rank);
    dataH = 0;
  }
  
  
  Handle<NrrdFieldConverterFieldAlgo> algo;
  
  if (!module_dynamic_compile(ci, algo)) return 0;
  
  
  //if( topology_ & STRUCTURED ) {
  //if (idim != 0) { // has been changed 
  if (topology_ == STRUCTURED) {
    ofield_handle = algo->execute( mHandle, dataH, idim, jdim, kdim, permute);
    
  } else  {
    ofield_handle = algo->execute( mHandle, dataH );
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
  rval->add_namespace("SCITeem");
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
  case 7:
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


} // End namespace Teem



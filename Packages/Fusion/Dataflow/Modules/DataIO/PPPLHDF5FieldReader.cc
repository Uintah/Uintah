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
 *  PPPLHDF5FieldReader.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   May 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Containers/Handle.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>

#include <Packages/Fusion/Dataflow/Modules/DataIO/PPPLHDF5FieldReader.h>

#include <sci_defs.h>


#include <sys/stat.h>

#ifdef HAVE_HDF5
#include "hdf5.h"
#endif

namespace Fusion {

using namespace SCIRun;

DECLARE_MAKER(PPPLHDF5FieldReader)

PPPLHDF5FieldReader::PPPLHDF5FieldReader(GuiContext *context)
  : Module("PPPLHDF5FieldReader", context, Source, "DataIO", "Fusion"),
    filename_(context->subVar("filename")),

    nDataSets_(context->subVar("ndatasets")),
    dataSet_  (context->subVar("dataset")),
    readAll_  (context->subVar("readall")),

    nDims_(context->subVar("ndims")),

    iDim_(context->subVar("i-dim")),
    jDim_(context->subVar("j-dim")),
    kDim_(context->subVar("k-dim")),

    iStart_(context->subVar("i-start")),
    jStart_(context->subVar("j-start")),
    kStart_(context->subVar("k-start")),

    iCount_(context->subVar("i-count")),
    jCount_(context->subVar("j-count")),
    kCount_(context->subVar("k-count")),

    iStride_(context->subVar("i-stride")),
    jStride_(context->subVar("j-stride")),
    kStride_(context->subVar("k-stride")),
  
    iWrap_(context->subVar("i-wrap")),
    jWrap_(context->subVar("j-wrap")),
    kWrap_(context->subVar("k-wrap")),

    idim_(0),
    jdim_(0),
    kdim_(0),

    istart_(-1),
    jstart_(-1),
    kstart_(-1),

    icount_(-1),
    jcount_(-1),
    kcount_(-1),

    istride_(1),
    jstride_(1),
    kstride_(1),

    iwrap_(0),
    jwrap_(0),
    kwrap_(0),

    fGeneration_(-1)
{
}

PPPLHDF5FieldReader::~PPPLHDF5FieldReader() {
}

void PPPLHDF5FieldReader::execute() {

#ifdef HDF5
  bool updateAll  = false;
  bool updateFile = false;

  int ndims;

  string new_filename(filename_.get());
  
  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if (stat(new_filename.c_str(), &buf)) {
    error( string("File not found ") + new_filename );
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif

  if( new_filename         != old_filename_ || 
      new_filemodification != old_filemodification_) {
    updateFile = true;
  }

  if( readall_ != readAll_.get() ||
      dataset_ != dataSet_.get() ||
      
      istart_ != iStart_.get() ||
      jstart_ != jStart_.get() ||
      kstart_ != kStart_.get() ||

      icount_ != iCount_.get() ||
      jcount_ != jCount_.get() ||
      kcount_ != kCount_.get() ||

      istride_ != iStride_.get() ||
      jstride_ != jStride_.get() ||
      kstride_ != kStride_.get() ||

      iwrap_ != iWrap_.get() ||
      jwrap_ != jWrap_.get() ||
      kwrap_ != kWrap_.get() ) {

    readall_ = readAll_.get();
    dataset_ = dataSet_.get();
      
    istart_ = iStart_.get();
    jstart_ = jStart_.get();
    kstart_ = kStart_.get();

    icount_ = iCount_.get();
    jcount_ = jCount_.get();
    kcount_ = kCount_.get();

    istride_ = iStride_.get();
    jstride_ = jStride_.get();
    kstride_ = kStride_.get();
 
    iwrap_ = iWrap_.get();
    jwrap_ = jWrap_.get();
    kwrap_ = kWrap_.get();
 
    updateAll = true;
  }


  if( updateFile || updateAll )
  {
    remark( "Reading the file " +  new_filename );

    old_filemodification_ = new_filemodification;
    old_filename_         = new_filename;

    herr_t  status;
 
    /* Open the file using default properties. */
    hid_t file_id = H5Fopen(new_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the grid group in the file. */
    hid_t g_id = H5Gopen(file_id, "coordinates");

    /* Open the coordinate dataset in the file. */
    hid_t ds_id = H5Dopen(g_id, "values"  );


    int rank;

    double* grid = NULL;
    double* data = NULL;

    {
      /* Open the coordinate space in the file. */
      hid_t file_space_id = H5Dget_space( ds_id );
    
      /* Get the rank (number of dims) in the space. */
      ndims = H5Sget_simple_extent_ndims(file_space_id);

      hsize_t *dims = new hsize_t[ndims];

      /* Get the dims in the space. */
      int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

      idim_ = dims[0];
      jdim_ = dims[1];
      kdim_ = dims[2];

      hssize_t *start = new hssize_t[ndims];
      hsize_t *stride = new hsize_t[ndims];
      hsize_t *count = new hsize_t[ndims];
      hsize_t *block = new hsize_t[ndims];

      /* Sample every th data point. */
      start[0]  = istart_;
      stride[0] = istride_;
      count[0]  = icount_;
      block[0]  = 1;

      start[1]  = jstart_;
      stride[1] = jstride_;
      count[1]  = jcount_;
      block[1]  = 1;

      start[2]  = kstart_;
      stride[2] = kstride_;
      count[2]  = kcount_;
      block[2]  = 1;

      start[3]  = 0;
      stride[3] = 1;
      count[3]  = dims[3];
      block[3]  = 1;
  
      int cc = icount_ * jcount_ * kcount_ * dims[3];
  
      status = H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET,
				   start, stride, count, block);

      hid_t mem_space_id = H5Screate_simple(ndims, count, NULL );

      for( int d=0; d<ndims; d++ ) {
	start[d] = 0;
	stride[d] = 1;
      }

      status = H5Sselect_hyperslab(mem_space_id, H5S_SELECT_SET,
				   start, stride, count, block);

      grid = new double[cc];

      status = H5Dread(ds_id, H5T_NATIVE_DOUBLE,
		       mem_space_id, file_space_id, H5P_DEFAULT, 
		       grid);

      /* Terminate access to the data space. */
      status = H5Sclose(file_space_id);
      /* Terminate access to the data space. */
      status = H5Sclose(mem_space_id);

      delete dims;
      delete start;
      delete stride;
      delete count;
      delete block;
    }

    /* Terminate access to the dataset. */
    status = H5Dclose(ds_id);
    /* Terminate access to the group. */ 
    status = H5Gclose(g_id);


  
    /* Open the node data group in the file. */
    g_id = H5Gopen(file_id, "node_data[0]");

    /* Open the coordinate dataset in the file. */
    ds_id = H5Dopen(g_id, "values"  );

    {
      /* Open the coordinate space in the file. */
      hid_t file_space_id = H5Dget_space( ds_id );

      /* Get the rank (number of dims) in the space. */
      ndims = H5Sget_simple_extent_ndims(file_space_id);

      hsize_t *dims = new hsize_t[ndims];

      /* Get the dims in the space. */
      int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

      if( ndims == 3 )
	rank = 1;
      else
	rank = dims[ndims-1];

      if( rank != 1 && rank != 3 ) {
	error( "Bad data rank, must be 1 or 3." );
	return;
      }

      hssize_t *start = new hssize_t[ndims];
      hsize_t *stride = new hsize_t[ndims];
      hsize_t *count = new hsize_t[ndims];
      hsize_t *block = new hsize_t[ndims];

      start[0]  = istart_;
      stride[0] = istride_;
      count[0]  = icount_;
      block[0]  = 1;

      start[1]  = jstart_;
      stride[1] = jstride_;
      count[1]  = jcount_;
      block[1]  = 1;

      start[2]  = kstart_;
      stride[2] = kstride_;
      count[2]  = kcount_;
      block[2]  = 1;

      int cc = icount_ * jcount_ * kcount_;

      status = H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET,
				   start, stride, count, block);

      hid_t mem_space_id = H5Screate_simple (ndims, count, NULL );

      for( int d=0; d<ndims; d++ ) {
	start[d] = 0;
	stride[d] = 1;
      }

      status = H5Sselect_hyperslab(mem_space_id, H5S_SELECT_SET,
				   start, stride, count, block);

      data = new double[cc];

      status = H5Dread(ds_id, H5T_NATIVE_DOUBLE,
		       mem_space_id, file_space_id, H5P_DEFAULT, 
		       data);

      /* Terminate access to the data space. */ 
      status = H5Sclose(file_space_id);
      /* Terminate access to the data space. */
      status = H5Sclose(mem_space_id);

      if( idim_ != dims[0] ||
	  jdim_ != dims[1] ||
	  kdim_ != dims[2] ) {
	error( "Grid and data do not have the same number of elements. " );
	return;
      }

      delete dims;
      delete start;
      delete stride;
      delete count;
      delete block;
    }

    /* Terminate access to the dataset. */
    status = H5Dclose(ds_id);
    /* Terminate access to the group. */ 
    status = H5Gclose(g_id);


    /* Terminate access to the group. */ 
    status = H5Fclose(file_id);


    remark( "Creating the field." );

    // 3D StructHexVolField
    if( icount_ > 1 && jcount_ > 1 && kcount_ > 1 ) {

      // Create the grid and scalar data matrices.
      StructHexVolMesh *shvm = scinew StructHexVolMesh(icount_+iwrap_,
						       jcount_+jwrap_,
						       kcount_+kwrap_);

      if( rank == 1 ) {
	// Now after the mesh has been created, create the field.
	StructHexVolField< double > *pfield =
	  scinew StructHexVolField<double>(shvm, Field::NODE);
	pHandle_ = pfield;
      } else if( rank == 3 ) {
	// Now after the mesh has been created, create the field.
	StructHexVolField< vector<double> > *pfield =
	  scinew StructHexVolField< vector<double> >(shvm, Field::NODE);
	pHandle_ = pfield;
      }
      else {
	error( "Bad data rank." );
	return;
      }

      // 2D StructQuadSurf
    } else if( icount_ == 1 || jcount_ == 1 || kcount_ == 1 ) {

      StructQuadSurfMesh *sqsm;
      
      if( kcount_ == 1 )
	sqsm = scinew StructQuadSurfMesh(icount_+iwrap_, jcount_+jwrap_);
      else if( jcount_ == 1 )
	sqsm = scinew StructQuadSurfMesh(icount_+iwrap_, kcount_+kwrap_);
      else if( icount_ == 1 )
	sqsm = scinew StructQuadSurfMesh(jcount_+jwrap_, kcount_+kwrap_);
      
      if( rank == 1 ) {
	// Now after the mesh has been created, create the field.
	StructQuadSurfField< double > *pfield =
	  scinew StructQuadSurfField< double >(sqsm, Field::NODE);
      	pHandle_ = pfield;
      } else if( rank == 3 ) {
	// Now after the mesh has been created, create the field.
	StructQuadSurfField< vector< double> > *pfield =
	  scinew StructQuadSurfField< vector<double> >(sqsm, Field::NODE);
      	pHandle_ = pfield;
      }
      else {
	error( "Bad data rank." );
	return;
      }
    } else {
      error( "Dimensions do not make sense." );
      return;
    }

    const TypeDescription *ftd = pHandle_->get_type_description(0);
    const TypeDescription *ttd = pHandle_->get_type_description(1);

    CompileInfoHandle ci = PPPLHDF5FieldReaderAlgo::get_compile_info(ftd, ttd);

    Handle<PPPLHDF5FieldReaderAlgo> algo;

    if (!module_dynamic_compile(ci, algo)) return;

    algo->execute(pHandle_,
		  icount_, jcount_, kcount_,
		  iwrap_, jwrap_, kwrap_, grid, data );

    delete grid;
    delete data;
  }
  else {
    remark( "Already read the file " +  new_filename );
  }

  // Get a handle to the output field port.
  if( pHandle_.get_rep() ) {
    FieldOPort *ofield_port = 
      (FieldOPort *) get_oport("Output Field");
    
    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( pHandle_ );
  }
#else

  error( "No HDF5 availible." );

#endif
}

void PPPLHDF5FieldReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("PPPLHDF5FieldReader needs a minor command");
    return;
  }

  if (args[1] == "update_file") {
#ifdef HAVE_HDF5
    int ndatasets, ndims;

    string new_filename(filename_.get());
  
    // Read the status of this file so we can compare modification timestamps
    struct stat buf;
    if (stat(new_filename.c_str(), &buf)) {
      error( string("File not found ") + new_filename );
      return;
    }

    // If we haven't read yet, or if it's a new filename, 
    //  or if the datestamp has changed -- then read...
#ifdef __sgi
    time_t new_filemodification = buf.st_mtim.tv_sec;
#else
    time_t new_filemodification = buf.st_mtime;
#endif

    if( new_filename         != old_filename_ || 
	new_filemodification != old_filemodification_) {

      herr_t  status;

      /* Open the file using default properties. */
      hid_t file_id = H5Fopen(new_filename.c_str(),
			      H5F_ACC_RDONLY, H5P_DEFAULT);


      /* Open the top level group in the file. */
      hid_t g_id = H5Gopen(file_id, "//");

      /* Open the number of data nodes in the file. */
      hid_t a_id = H5Aopen_name(g_id, "nnode_data");

      status = H5Aread( a_id, H5T_NATIVE_INT, &ndatasets );

      /* Terminate access to the attribute. */ 
      status = H5Aclose(a_id);
      /* Terminate access to the group. */ 
      status = H5Gclose(g_id);



      /* Open the grid group in the file. */
      g_id = H5Gopen(file_id, "coordinates");

      /* Open the coordinate dataset in the file. */
      hid_t ds_id = H5Dopen(g_id, "values"  );

      /* Open the coordinate space in the file. */
      hid_t file_space_id = H5Dget_space( ds_id );
    
      /* Get the rank (number of dims) in the space. */
      ndims = H5Sget_simple_extent_ndims(file_space_id);

      hsize_t *dims = new hsize_t[ndims];

      /* Get the dims in the space. */
      int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

      /* Terminate access to the data space. */ 
      status = H5Sclose(file_space_id);
      /* Terminate access to the dataset. */
      status = H5Dclose(ds_id);
      /* Terminate access to the group. */ 
      status = H5Gclose(g_id);
      /* Terminate access to the group. */ 
      status = H5Fclose(file_id);

      idim_ = dims[0];
      jdim_ = dims[1];
      kdim_ = dims[2];
      ndims--;
      delete dims;

      // Check to see if the dimensions have changed.
      if( ndims != nDims_.get() ||
	  idim_ != iDim_.get()  ||
	  jdim_ != jDim_.get()  ||
	  kdim_ != kDim_.get() ) {
    
	// Update the dims in the GUI.
	ostringstream str;
	str << id << " set_size " << ndatasets << " ";
	str << ndims << " " << idim_ << " " << jdim_ << " " << kdim_;
    
	gui->execute(str.str().c_str());
      }
    }
#else

  error( "No HDF5 availible." );

#endif
  } else {
    Module::tcl_command(args, userdata);
  }
}

CompileInfoHandle
PPPLHDF5FieldReaderAlgo::get_compile_info(const TypeDescription *ftd,
					  const TypeDescription *ttd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("PPPLHDF5FieldReaderAlgoT");
  static const string base_class_name("PPPLHDF5FieldReaderAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + "," +
		       ttd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("Fusion");
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace Fusion

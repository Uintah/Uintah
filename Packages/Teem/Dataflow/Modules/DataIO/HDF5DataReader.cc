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
 *  HDF5DataReader.cc:
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

#include <Core/Containers/Array1.h>

#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Packages/Teem/Dataflow/Modules/DataIO/HDF5DataReader.h>
#include <Packages/Teem/Core/Datatypes/NrrdData.h>

#include <sci_defs.h>

#include <sys/stat.h>

#include <fstream>

#ifdef HAVE_HDF5
#include "hdf5.h"
#include "HDF5Dump.h"
#endif

namespace SCITeem {

using namespace SCIRun;

DECLARE_MAKER(HDF5DataReader)

HDF5DataReader::HDF5DataReader(GuiContext *context)
  : Module("HDF5DataReader", context, Source, "DataIO", "Teem"),
    filename_(context->subVar("filename")),
    datasets_(context->subVar("datasets")),
    dumpname_(context->subVar("dumpname")),

    nDims_(context->subVar("ndims")),

    mergeData_(context->subVar("mergeData")),
    assumeSVT_(context->subVar("assumeSVT")),

    error_(false)
{
  for( int ic=0; ic<MAX_DIMS; ic++ ) {
    char idx[16];

    sprintf( idx, "%d-dim", ic );
    gDims_   .push_back(new GuiInt(context->subVar(idx)) );
    sprintf( idx, "%d-start", ic );
    gStarts_ .push_back(new GuiInt(context->subVar(idx)) );
    sprintf( idx, "%d-count", ic );
    gCounts_ .push_back(new GuiInt(context->subVar(idx)) );
    sprintf( idx, "%d-stride", ic );
    gStrides_.push_back(new GuiInt(context->subVar(idx)) );

    dims_[ic] = 0;
    starts_[ic] =  0;
    counts_[ic] =  0;
    strides_[ic] = 1;
  }
}

HDF5DataReader::~HDF5DataReader() {
}



// Allows nrrds to join along tuple i.e. scalar and vector sets can be joined,
// or allows for a multiple identical nrrds to assume a time series, 
// and be joined along a new time axis. 
bool
HDF5DataReader::is_mergeable(Nrrd* n1, Nrrd* n2) const
{
  if (n1->type != n2->type) return false;
  if (n1->dim != n2->dim)  return false;
  for (int i = 1; i < n1->dim; i++) {
    if (n1->axis[i].size != n2->axis[i].size) return false;
  }

  return true;
}


void HDF5DataReader::execute() {

#ifdef HAVE_HDF5
  bool updateAll  = false;
  bool updateFile = false;

  filename_.reset();
  datasets_.reset();

  string new_filename(filename_.get());
  string new_datasets(datasets_.get());

  vector< string > paths;
  vector< string > datasets;

  if( new_filename.length() == 0 || new_datasets.length() == 0 )
    return;

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
      new_filemodification != old_filemodification_ || 
      new_datasets         != old_datasets_ ) {

    old_filemodification_ = new_filemodification;
    old_filename_         = new_filename;
    old_datasets_         = new_datasets;

    updateFile = true;
  }
  // get all the actual values from gui.
  reset_vars();

  if( mergedata_ != mergeData_.get() ||
      assumesvt_ != assumeSVT_.get() ) {

    mergedata_ = mergeData_.get();
    assumesvt_ = assumeSVT_.get();

    updateAll = true;
  }

  for( int ic=0; ic<MAX_DIMS; ic++ ) {

    if( starts_ [ic] != gStarts_ [ic]->get() ||
	counts_ [ic] != gCounts_ [ic]->get() ||
	strides_[ic] != gStrides_[ic]->get() ) {

      starts_ [ic] = gStarts_ [ic]->get();
      counts_ [ic] = gCounts_ [ic]->get();
      strides_[ic] = gStrides_[ic]->get();
    
      updateAll = true;
    }

    if( dims_[ic] != gDims_[ic]->get() ) {
      dims_[ic] = gDims_[ic]->get();
      updateFile = true;
    }

    if( starts_[ic] + (counts_[ic]-1) * strides_[ic] >= dims_[ic] ) {
    
      error( "Data selection exceeds bounds." );
      error( "Decrease the start or count or increase the stride." );
      return;
    }
  }


  if( error_ ||
      updateFile ||
      updateAll )
  {
    error_ = false;

    vector< string > paths;
    vector< string > datasets;

    parseDatasets( new_datasets, paths, datasets );

    vector< vector<NrrdDataHandle> > nHandles;
    
    for( unsigned int ic=0; ic<paths.size(); ic++ ) {

      NrrdDataHandle handle = readDataset(new_filename, paths[ic], 
					  datasets[ic]);

      bool inserted = false;
      vector<vector<NrrdDataHandle> >::iterator iter = nHandles.begin();
      while (iter != nHandles.end()) {

	vector<NrrdDataHandle> &vec = *iter;
	++iter;

	if (vec.size() == 0) {
	  vec.push_back(handle);
	  inserted = true;
	  break;
	} else if(is_mergeable(handle->nrrd, vec[0]->nrrd)) {
	  vec.push_back(handle);
	  inserted = true;
	  break;
	}
      }

      if (! inserted) {
	vector<NrrdDataHandle> nrrdSet;
	nrrdSet.push_back( handle );
	nHandles.push_back( nrrdSet );
      }
    }

    // Merge the like datatypes together.
    if( mergedata_ ) {

      vector<vector<NrrdDataHandle> >::iterator iter = nHandles.begin();
      while (iter != nHandles.end()) {

	vector<NrrdDataHandle> &vec = *iter;
	++iter;

	if (vec.size() > 1) {
	  
	  NrrdData* onrrd = new NrrdData(true);
	  string new_label("");
	  string axis_label("");

	  bool time_axis = true;
	  // check if this is a time axis merge or a tuple axis merge.
	  vector<Nrrd*> join_me;
	  vector<NrrdDataHandle>::iterator niter = vec.begin();
	  NrrdDataHandle first = *niter;
	  string first_tlab(first->nrrd->axis[0].label);
	  new_label = first_tlab;
	  string first_tuple_types = first->concat_tuple_types();
	  cout << "the tuple types are: " << first_tuple_types << endl;
	  ++niter;
	  join_me.push_back(first->nrrd);
	  while (niter != vec.end()) {
	    NrrdDataHandle n = *niter;
	    ++niter;
	    join_me.push_back(n->nrrd);
	    string tlab(n->nrrd->axis[0].label);
	    string types = n->concat_tuple_types();
	    if (types != first_tuple_types) time_axis = false;

	    new_label += string(",") + tlab;

	  }	  
	  int axis = 0; // tuple
	  int incr = 0; // tuple case.
	  if (time_axis) {
	    axis = join_me[0]->dim;
	    incr = 1;
	  }
	  onrrd->nrrd = nrrdNew();
	  if (nrrdJoin(onrrd->nrrd, &join_me[0], join_me.size(), axis, incr)) {
	      char *err = biffGetDone(NRRD);
	      error(string("Join Error: ") +  err);
	      free(err);
	      error_ = true;
	      return;
	    }
	  

	  if (time_axis) {
	    onrrd->nrrd->axis[axis].label = "Time";
	    // remove all numbers from name
	    string s(join_me[0]->axis[0].label);
	    string new_label;

	    const string nums("0123456789");
	      
	    //cout << "checking in_name " << s << endl;
	    // test against valid char set.

	    for(string::size_type i = 0; i < s.size(); i++) {
	      bool in_set = false;
	      for (unsigned int c = 0; c < nums.size(); c++) {
		if (s[i] == nums[c]) {
		  in_set = true;
		  break;
		}
	      }

	      if (in_set) { new_label.push_back('X' ); }
	      else        { new_label.push_back(s[i]); }

	    }
	    onrrd->nrrd->axis[0].label = strdup(new_label.c_str());
	  } else {
	    // Take care of tuple axis label.
	    onrrd->nrrd->axis[0].label = strdup(new_label.c_str());
	  }

	  // clear the nrrds;
	  vec.clear();
	  vec.push_back(onrrd);
	}
      }
    }

    int cc = 0;

    for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {
      for( unsigned int jc=0; jc<nHandles[ic].size(); jc++ ) {
	nHandles_[cc] = nHandles[ic][jc];

	++cc;
	if( cc == MAX_PORTS ) {
	  warning( "Maximum number of ports reached" );
	  break;
	}
      }

      if( cc == MAX_PORTS )
	break;
    }

    for( int ic=cc; ic<MAX_PORTS; ic++ )
      nHandles_[ic] = NULL;
  }
  else {
    remark( "Already read the file " +  new_filename );
  }

  for( int ic=0; ic<MAX_PORTS; ic++ ) {
    // Get a handle to the output double port.
    if( nHandles_[ic].get_rep() ) {

      char portNumber[4];
      sprintf( portNumber, "%d", ic+1 );

      string portName = string("Output ") +
	string(portNumber) +
	string( " Nrrd" );

      
      NrrdOPort *ofield_port = 
	(NrrdOPort *) get_oport(portName);
    
      if (!ofield_port) {
	error("Unable to initialize "+name+"'s " + portName + " oport\n");
	return;
      }

      // Send the data downstream
      ofield_port->send( nHandles_[ic] );
    }
  }
#else  
  error( "No HDF5 availible." );
  
#endif
}


void HDF5DataReader::parseDatasets( string new_datasets,
				   vector<string>& paths,
				   vector<string>& datasets )
{
  int open = 0;

  std::string::size_type cc = 0;
  std::string::size_type bb = 0;
  std::string::size_type ee = 0;

  string path, dataset;

  while( cc < new_datasets.length() ) {

    bb = ee = cc;

    if( new_datasets[bb] == '{' ) {
      // Multiple datasets.

      open = 1;
      ee = bb + 1;

      while( open && ee < new_datasets.length() ) {
	ee++;

	if( new_datasets[ee] == '{' )
	  open++;
	else if( new_datasets[ee] == '}' )
	  open--;
      }

      path = new_datasets.substr( bb+1, ee-bb-1);

      cc = ee + 2;

    } else {
      // Single Dataset
      path = new_datasets;

      cc = new_datasets.length();
    }

    // Get the dataset name.
    std::string::size_type last = path.find_last_of( " " );

    string dataset( path.substr( last+1, path.length()-last) );

    // Remove the dataset name from the path.
    path.erase( last, path.length()-last);

    // Remove the first space.
    path.erase( 1, 1 );

    // Replace the other spaces with a slash.
    while( (last = path.find( " " )) != std::string::npos )
      path.replace( last, 1, "/" );
    
    // Remove the rest of the braces.
    while( (last = path.find( "{" )) != std::string::npos )
      path.erase( last, 1 );
    
    while( (last = path.find( "}" )) != std::string::npos )
      path.erase( last, 1 );

    paths.push_back( path );
    datasets.push_back( dataset );
  }
}

vector<int> HDF5DataReader::getDatasetDims( string filename,
					    string group,
					    string dataset ) {
  vector< int > idims;
  
#ifdef HAVE_HDF5
  herr_t status;

  /* Open the file using default properties. */
  hid_t file_id, ds_id, g_id, file_space_id;

  if( (file_id = H5Fopen(filename.c_str(),
			 H5F_ACC_RDONLY, H5P_DEFAULT)) < 0 ) {
    error( "Error opening file. " );
    return idims;
  }

  /* Open the group in the file. */
  if( (g_id = H5Gopen(file_id, group.c_str())) < 0 ) {
    error( "Error opening group. " );
    return idims;
  }

  /* Open the dataset in the file. */
  if( (ds_id = H5Dopen(g_id, dataset.c_str())) < 0 ) {
    error( "Error opening dataset. " );
    return idims;
  }

  /* Open the coordinate space in the file. */
  if( (file_space_id = H5Dget_space( ds_id )) < 0 ) {
    error( "Error getting file space. " );
    return idims;
  }
    
  /* Get the rank (number of dims) in the space. */
  int ndims = H5Sget_simple_extent_ndims(file_space_id);

  if (H5Sis_simple(file_space_id)) {
    if (ndims == 0) {
      /* scalar dataspace */
      idims.push_back( 1 );

    } else {
      /* simple dataspace */
      hsize_t *dims = new hsize_t[ndims];

      /* Get the dims in the space. */
      int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

      if( ndim != ndims ) {
	error( "Data dimensions not match. " );
	return idims;
      }

      for( int ic=0; ic<ndims; ic++ )
	idims.push_back( dims[ic] );
      
      delete dims;
    }
  }

  /* Terminate access to the data space. */ 
  if( (status = H5Sclose(file_space_id)) < 0 ) {
    error( "Error closing file space. " );
  }
  /* Terminate access to the dataset. */
  if( (status = H5Dclose(ds_id)) < 0 ) {
    error( "Error closing data set. " );
  }
  /* Terminate access to the group. */ 
  if( (status = H5Gclose(g_id)) < 0 ) {
    error( "Error closing group. " );
  }
  /* Terminate access to the group. */ 
  if( (status = H5Fclose(file_id)) < 0 ) {
    error( "Error closing file. " );
  }
#endif

  return idims;
}


NrrdDataHandle HDF5DataReader::readDataset( string filename,
					   string group,
					   string dataset ) {
#ifdef HAVE_HDF5
  void *data = NULL;

  herr_t  status;
 
  /* Open the file using default properties. */
  hid_t file_id, g_id, ds_id, file_space_id;

  if( (file_id = H5Fopen(filename.c_str(),
			 H5F_ACC_RDONLY, H5P_DEFAULT)) < 0 ) {
    error( "Error opening file. " );
    return NULL;
  }

  /* Open the group in the file. */
  if( (g_id = H5Gopen(file_id, group.c_str())) < 0 ) {
    error( "Error opening group. " );
    return NULL;
  }

  /* Open the dataset in the file. */
  if( (ds_id = H5Dopen(g_id, dataset.c_str())) < 0 ) {
    error( "Error opening data space. " );
    return NULL;
  }

  /* Open the coordinate space in the file. */
  if( (file_space_id = H5Dget_space( ds_id )) < 0 ) {
    error( "Error opening file space. " );
    return NULL;
  }
    
  hid_t type_id = H5Dget_type(ds_id);

  hid_t mem_type_id;

  unsigned int nrrd_type;

  switch (H5Tget_class(type_id)) {
  case H5T_INTEGER:
    // Integer
    mem_type_id = H5T_NATIVE_INT;
    nrrd_type = get_nrrd_type<int>();
    break;

  case H5T_FLOAT:
    if (H5Tequal(type_id, H5T_IEEE_F32BE) ||
	H5Tequal(type_id, H5T_IEEE_F32LE) ||
	H5Tequal(type_id, H5T_NATIVE_FLOAT)) {
      // Float
      mem_type_id = H5T_NATIVE_FLOAT;
      nrrd_type = get_nrrd_type<float>();

    } else if (H5Tequal(type_id, H5T_IEEE_F64BE) ||
	       H5Tequal(type_id, H5T_IEEE_F64LE) ||
	       H5Tequal(type_id, H5T_NATIVE_DOUBLE) ||
	       H5Tequal(type_id, H5T_NATIVE_LDOUBLE)) {
      // Double
      mem_type_id = H5T_NATIVE_DOUBLE;
      nrrd_type = get_nrrd_type<double>();

    } else {
      error("Undefined HDF5 float");
      error_ = true;
      return NULL;
    }
    break;
  case H5T_COMPOUND:
    error("Compound HDF5 data types can be converted into Nrrds.");
    error_ = true;
    return NULL;
    break;

  default:
    error("Unknown or unsupported HDF5 data type");
    error_ = true;
    return NULL;
  }

  hsize_t size = H5Tget_size(type_id);

  if( H5Tget_size(type_id) > H5Tget_size(mem_type_id) )
    size = H5Tget_size(type_id);
  else
    size = H5Tget_size(mem_type_id);

  if( size == 0 ) {
    error( "Null data size. " );
    return NULL;
  }

  H5Tclose(type_id);
 
  /* Get the rank (number of dims) in the space. */
  int ndims = H5Sget_simple_extent_ndims(file_space_id);

  hsize_t *count;
  hsize_t *dims;

  if (H5Sis_simple(file_space_id)) {
    if (ndims == 0) {
      /* scalar dataspace */
      ndims = 1;
      dims = new hsize_t[ndims];
      count = new hsize_t[ndims];

      dims[0] = 1;
      count[0] = 1;

      if( (data = new char[size]) == NULL ) {
	error( "Can not allocate enough memory for the data" );
	error_ = true;
	return NULL;
      }

      if( (status = H5Dread(ds_id, mem_type_id,
			    H5S_ALL, H5S_ALL, H5P_DEFAULT, 
			    data)) < 0 ) {
	error( "Error reading dataset." );
	error_ = true;
	return NULL;
      }
    } else {
      /* simple dataspace */
      dims = new hsize_t[ndims];
      count = new hsize_t[ndims];

      /* Get the dims in the space. */
      int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

      if( ndim != ndims ) {
	error( "Data dimensions not match. " );
	error_ = true;
	return NULL;
      }

      for( int ic=0; ic<nDims_.get(); ic++ ) {
	if( (unsigned int) dims_[ic] != dims[ic] ) {
	  error( "Data do not have the same number of elements. " );
	  error_ = true;
	  return NULL;
	}
      }


      hssize_t *start = new hssize_t[ndims];
      hsize_t *stride = new hsize_t[ndims];
      hsize_t *block = new hsize_t[ndims];

      for( int ic=0; ic<nDims_.get(); ic++ ) {
	start[ic]  = starts_[ic];
	stride[ic] = strides_[ic];
	count[ic]  = counts_[ic];
	block[ic]  = 1;
      }

      for( int ic=nDims_.get(); ic<ndims; ic++ ) {
	start[ic]  = 0;
	stride[ic] = 1;
	count[ic]  = dims[ic];
	block[ic]  = 1;
      }

      if( (status = H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET,
					start, stride, count, block)) < 0 ) {
	error( "Can not select data slab requested." );
	error_ = true;
	return NULL;
      }

      hid_t mem_space_id = H5Screate_simple (ndims, count, NULL );

      for( int d=0; d<ndims; d++ ) {
	start[d] = 0;
	stride[d] = 1;
      }

      if( (status = H5Sselect_hyperslab(mem_space_id, H5S_SELECT_SET,
					start, stride, count, block)) < 0 ) {
	error( "Can not select memory for the data slab requested." );
	error_ = true;
	return NULL;
      }

      for( int ic=0; ic<ndims; ic++ )
	size *= count[ic];

      if( (data = new char[size]) == NULL ) {
	error( "Can not allocate enough memory for the data" );
	error_ = true;
	return NULL;
      }

      if( (status = H5Dread(ds_id, mem_type_id,
			    mem_space_id, file_space_id, H5P_DEFAULT, 
			    data)) < 0 ) {
	error( "Can not read the data slab requested." );
	error_ = true;
	return NULL;
      }

      /* Terminate access to the data space. */
      if( (status = H5Sclose(mem_space_id)) < 0 ) {
	error( "Can not cloase the memory data slab requested." );
	error_ = true;
	return NULL;
      }

      delete start;
      delete stride;
      delete block;
    }
  }

  /* Terminate access to the data space. */ 
  if( (status = H5Sclose(file_space_id)) < 0 )
    error( "Can not cloase file space." );

  /* Terminate access to the dataset. */
  if( (status = H5Dclose(ds_id)) < 0 )
    error( "Can not cloase file space." );

  /* Terminate access to the group. */ 
  if( (status = H5Gclose(g_id)) < 0 )
    error( "Can not cloase file space." );

  /* Terminate access to the file. */ 
  if( (status = H5Fclose(file_id)) < 0 )
    error( "Can not cloase file space." );

  string tuple_type_str(":Scalar");
  int sink_size = 1;
  NrrdData *nout = scinew NrrdData(false);

  switch(ndims) {
  case 1: 
    nrrdWrap(nout->nrrd, data,
	     nrrd_type, ndims+1, sink_size, (unsigned int) count[0]);
    nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		nrrdCenterNode);
    break;

  case 2: 
    {
      // If the user asks us to assume vector or tensor data, the
      // assumption is based on the size of the last dimension.
      int sz_last_dim = 1;
      if (assumesvt_) { sz_last_dim = count[1];} 
      
      switch (sz_last_dim) {
      case 3: // Vector data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		 (unsigned int) count[0], (unsigned int) count[1]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);
	tuple_type_str = ":Vector";
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		 (unsigned int) count[0], (unsigned int) count[1]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);	
	break;
      };
    }
    break;

  case 3: 
    {
      // If the user asks us to assume vector or tensor data, the
      // assumption is based on the size of the last dimension.
      int sz_last_dim = 1;
      if (assumesvt_) { sz_last_dim = count[2];} 
      
      switch (sz_last_dim) {
      case 3: // Vector data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	tuple_type_str = ":Vector";
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	break;
      };
    }
    break;

  case 4: 
    {
      // If the user asks us to assume vector or tensor data, the
      // assumption is based on the size of the last dimension.
      int sz_last_dim = 1;
      if (assumesvt_) { sz_last_dim = count[3]; } 

      switch (sz_last_dim) {
      case 3: // Vector data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode);
	tuple_type_str = ":Vector";
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode);
	break;
      };
    }
    break;

  case 5: 
    {
      // If the user asks us to assume vector or tensor data, the
      // assumption is based on the size of the last dimension.
      int sz_last_dim = 1;
      if (assumesvt_) { sz_last_dim = count[4];} 
      
      switch (sz_last_dim) {
      case 3: // Vector data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);
	tuple_type_str = ":Vector";
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);
	break;
      };
    }

    break;

  case 6: 
        {
      // If the user asks us to assume vector or tensor data, the
      // assumption is based on the size of the last dimension.
      int sz_last_dim = 1;
      if (assumesvt_) { sz_last_dim = count[5];} 
      
      switch (sz_last_dim) {
      case 3: // Vector data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4], dims[5]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	tuple_type_str = ":Vector";
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4], dims[5]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4], dims[5]);
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	break;
      };
    }
    break;
  }

  string sink_label = group + "-" + dataset + tuple_type_str;

  // Remove all of the tcl special characters.
  std::string::size_type pos;
  while( (pos = sink_label.find("/")) != string::npos )
    sink_label.replace( pos, 1, "-" );
  while( (pos = sink_label.find("[")) != string::npos )
    sink_label.replace( pos, 1, "_" );
  while( (pos = sink_label.find("]")) != string::npos )
    sink_label.erase( pos, 1 );

  nout->nrrd->axis[0].label = strdup(sink_label.c_str());

  for( int ic=0; ic<ndims; ic++ ) {
    char tmpstr[16];

    sprintf( tmpstr, "%d", (int) (count[ic]) );
    nout->nrrd->axis[ic+1].label = strdup(tmpstr);
  }


  delete dims;
  delete count;

  return NrrdDataHandle(nout);
#else
  return NULL;
#endif
}

void HDF5DataReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("HDF5DataReader needs a minor command");
    return;
  }

  if (args[1] == "update_file") {
#ifdef HAVE_HDF5
    filename_.reset();
    string new_filename(filename_.get());

    if( new_filename.length() == 0 )
      return;

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

      // Get the file hierarchy for the tcl browser.
      std::string::size_type last = new_filename.find_last_of( "/" );

      if( last == string::npos )
	last = 0;
      else
	last++;

      string tmp_filename( "/tmp/" );
      tmp_filename.append( new_filename, last, new_filename.length()-last );
      tmp_filename.append( ".dump" );

      std::ofstream sPtr( tmp_filename.c_str() );

      if( !sPtr ) {
	error( string("Unable to open output file: ") + tmp_filename );
	return;
      }
  
      if( HDF5Dump_file( new_filename.c_str(), &sPtr ) < 0 ) {
	error( string("Could not create dump file: ") + tmp_filename );
	return;
      }

      sPtr.flush();
      sPtr.close();

      if (stat(tmp_filename.c_str(), &buf)) {
	error( string("Temporary dump file not found ") + tmp_filename );
	return;
      } 

      // Update the treeview in the GUI.
      ostringstream str;
      str << id << " build_tree " << tmp_filename;
      
      gui->execute(str.str().c_str());
      
      // Update the dims in the GUI.
      gui->execute(id + " set_size 0 {}");
    }
#else
    error( "No HDF5 availible." );
#endif

  } else if (args[1] == "update_selection") {
#ifdef HAVE_HDF5
    filename_.reset();
    datasets_.reset();
    string new_filename(filename_.get());
    string new_datasets(datasets_.get());

    if( new_filename.length() == 0 ||  new_datasets.length() == 0 )
      return;

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
	new_filemodification != old_filemodification_ ||
	new_datasets         != old_datasets_ ) {

      vector<string> paths;
      vector<string> datasets;

      parseDatasets( new_datasets, paths, datasets );

      unsigned long ndims = 0;

      for( int ic=0; ic<MAX_DIMS; ic++ )
	dims_[ic] = 1;

      for( unsigned int ic=0; ic<paths.size(); ic++ ) {

	vector<int> dims =
	  getDatasetDims( new_filename, paths[ic], datasets[ic] );

	if( ic == 0 ) {

	  ndims = dims.size();

	  for( unsigned int jc=0; jc<ndims && jc<MAX_DIMS; jc++ )
	    dims_[jc] = dims[jc];
	} else {

	  if( ndims > dims.size() )
	    ndims = dims.size();

	  for( unsigned int jc=0; jc<ndims && jc<MAX_DIMS; jc++ ) {
	    if( dims_[jc] != dims[jc] ) {
	      ndims = 0;
	      break;
	    }
	  }
	}
      }

      bool set = (ndims != (unsigned int) nDims_.get());

      if( !set ) {
	for( int ic=0; ic<MAX_DIMS; ic++ ) {
	  if( dims_[ic] != gDims_[ic]->get() ) {
	    set = true;
	  }
	}
      }

      // Check to see if the dimensions have changed.
      if( set ) {

	string dimstr( "{ " );

	for( unsigned int ic=0; ic<ndims && ic<MAX_DIMS; ic++ ) {
	  char dim[8];

	  sprintf( dim, " %d ", dims_[ic] );

	  dimstr += dim;
	}

	dimstr += " }";

	// Update the dims in the GUI.
	ostringstream str;
	str << id << " set_size " << ndims << " " << dimstr;
    
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

} // End namespace SCITeem

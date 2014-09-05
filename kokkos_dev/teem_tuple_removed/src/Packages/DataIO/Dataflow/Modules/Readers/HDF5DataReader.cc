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

#include <Core/Malloc/Allocator.h>

#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <Packages/DataIO/Dataflow/Modules/Readers/HDF5DataReader.h>

#include <sci_defs.h>

#include <sys/stat.h>

#include <fstream>
#include <algorithm>

#ifdef HAVE_HDF5
#include "hdf5.h"
#include "HDF5Dump.h"
#endif

namespace DataIO {

using namespace SCIRun;
using namespace SCITeem;

DECLARE_MAKER(HDF5DataReader)

HDF5DataReader::HDF5DataReader(GuiContext *context)
  : Module("HDF5DataReader", context, Source, "Readers", "DataIO"),
    selectable_min_(ctx->subVar("selectable_min")),
    selectable_max_(ctx->subVar("selectable_max")),
    selectable_inc_(ctx->subVar("selectable_inc")),
    range_min_(ctx->subVar("range_min")),
    range_max_(ctx->subVar("range_max")),
    playmode_(ctx->subVar("playmode")),
    dependence_(ctx->subVar("dependence")),
    current_(ctx->subVar("current")),
    delay_(ctx->subVar("delay")),
    inc_amount_(ctx->subVar("inc-amount")),
    inc_(1),
    execmode_("none"),
    last_input_(-1),
    last_output_(0),

    filename_(context->subVar("filename")),
    datasets_(context->subVar("datasets")),
    dumpname_(context->subVar("dumpname")),
    ports_(context->subVar("ports")),

    nDims_(context->subVar("ndims")),

    mergeData_(context->subVar("mergeData")),
    assumeSVT_(context->subVar("assumeSVT")),

    animate_(context->subVar("animate")),

    mergedata_(-1),
    assumesvt_(-1),

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



// Allows nrrds to join along tuple, scalar and vector sets can not be joined,
// or allows for a multiple identical nrrds to assume a time series, 
// and be joined along a new time axis. 
bool
HDF5DataReader::is_mergeable(NrrdDataHandle h1, NrrdDataHandle h2) const
{
  if( mergedata_ != 2 ) 
    if( (*((PropertyManager *) h1.get_rep()) !=
	 *((PropertyManager *) h2.get_rep())) ) return false;
  
  // FIX ME
  //cerr << "Tuple strings: " << h1->concat_tuple_types() << " " << h2->concat_tuple_types() << "\n";
  //if( h1->concat_tuple_types() != h2->concat_tuple_types() ) {
  //return false;
  //}

  Nrrd* n1 = h1->nrrd;
  Nrrd* n2 = h2->nrrd;

  if (n1->type != n2->type) return false;
  if (n1->dim  != n2->dim)  return false;

  // FIX ME
  //for (int i=1; i<n1->dim; i++) {
  int start = 0;
  if (n1->dim >= n2->dim)
    start = n2->dim;
  else
    start = n1->dim;
  // compare the last dimensions (in case scalar and vector)
  for (int i=start-1; i>=0; i--) {
    if (n1->axis[i].size != n2->axis[i].size) {
      return false;
    }
  }

  return true;
}


void HDF5DataReader::execute() {

#ifdef HAVE_HDF5
  bool updateAll   = false;
  bool updateFile  = false;

  filename_.reset();
  datasets_.reset();

  string new_filename(filename_.get());
  string new_datasets(datasets_.get());

  if( new_filename.length() == 0 || new_datasets.length() == 0 )
    return;

  // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
  struct stat64 buf;
  if (stat64(new_filename.c_str(), &buf)) {
#else
  struct stat buf;
  if (stat(new_filename.c_str(), &buf)) {
#endif
    error( string("Execute File not found ") + new_filename );
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

    tmp_filemodification_ = new_filemodification;
    tmp_filename_         = new_filename;
    tmp_datasets_         = new_datasets;

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

  vector< string > paths;
  vector< string > datasets;
  
  parseDatasets( new_datasets, paths, datasets );
  

  if( animate_.get() ) {

    vector< vector<string> > frame_paths;
    vector< vector<string> > frame_datasets;
    
    unsigned int nframes =
      parseAnimateDatasets( paths, datasets, frame_paths, frame_datasets);
    

    // If there is a current index matrix, use it.
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Current Index");

    if (!imatrix_port) {
      error("Unable to initialize iport 'Current Index'.");
      return;
    }

    MatrixHandle mHandle;
    if (imatrix_port->get(mHandle) && mHandle.get_rep()) {
      unsigned int which = (unsigned int) (mHandle->get(0, 0));

      if( 0 <= which && which <= frame_paths.size() ) {
	current_.set(which);
	current_.reset();
	
	ReadandSendData( new_filename, frame_paths[which],
			 frame_datasets[which], true, true, which );
      } else {
	error( "Input index is out of range" );
	return;
      }
    } else {

      if( nframes != selectable_max_.get() ) {
	ostringstream str;
	str << id << " update_animate_range 0 " << nframes-1;
	gui->execute(str.str().c_str());
      }

      animate_execute( new_filename, frame_paths, frame_datasets );
    }
  } else if( error_ ||
	     updateFile ||
	     updateAll )
  {
    error_ = false;

    ReadandSendData( new_filename, paths, datasets, true, true, -1 );

  } else {
    remark( "Already read the file " +  new_filename );

    for( unsigned int ic=0; ic<MAX_PORTS; ic++ ) {
      // Get a handle to the output double port.
      if( nHandles_[ic].get_rep() ) {

	char portNumber[4];
	sprintf( portNumber, "%d", ic );

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
  }


#else  
  error( "No HDF5 availible." );
  
#endif
}

void HDF5DataReader::ReadandSendData( string& filename,
				      vector< string >& paths,
				      vector< string >& datasets,
				      bool last,
				      bool cache,
				      int which ) {

  vector< vector<NrrdDataHandle> > nHandles;
  vector< vector<int> > ids;

  vector< int > ports;
    
  for( unsigned int ic=0; ic<paths.size(); ic++ ) {
    ports.push_back( -1 );

    NrrdDataHandle nHandle =
      readDataset(filename, paths[ic], datasets[ic]);

    if( nHandle != NULL ) {
      bool inserted = false;
      vector<vector<NrrdDataHandle> >::iterator iter = nHandles.begin();
      vector<vector<int> >::iterator iditer = ids.begin();
	
      while (iter != nHandles.end()) {

	vector<NrrdDataHandle> &vec = *iter;
	vector<int> &idvec = *iditer;

	++iter;
	++iditer;

	if(is_mergeable(nHandle, vec[0])) {
	  vec.push_back(nHandle);
	  idvec.push_back(ic);
	  inserted = true;
	  break;
	}
      }

      if (!inserted) {
	vector<NrrdDataHandle> nrrdSet;
	nrrdSet.push_back( nHandle );
	nHandles.push_back( nrrdSet );

	vector<int> idSet;
	idSet.push_back( ic );
	ids.push_back( idSet );
      }
    } else {
      error( "No handle for - " + paths[ic] + "/" + datasets[ic] );
      return;
    }
  }

  // Merge the like datatypes together.
  if( mergedata_ ) {

    vector<vector<NrrdDataHandle> >::iterator iter = nHandles.begin();
    while (iter != nHandles.end()) {

      vector<NrrdDataHandle> &vec = *iter;
      ++iter;

      if (mergedata_ == MERGE_TIME || vec.size() > 1) {
	  
	string new_label("");
	string axis_label("");

	// check if this is a time axis merge or a tuple axis merge.
	vector<Nrrd*> join_me;
	vector<NrrdDataHandle>::iterator niter = vec.begin();

	NrrdDataHandle n = *niter;
	++niter;
	join_me.push_back(n->nrrd);
	// FIX ME
	new_label = n->nrrd->axis[0].label;

	while (niter != vec.end()) {
	  NrrdDataHandle n = *niter;
	  ++niter;
	  join_me.push_back(n->nrrd);
	  // FIX ME
	  new_label += string(",") + n->nrrd->axis[0].label;
	}	  

	if (join_me.size() == 3 || join_me.size() == 6) {
	  NrrdData* onrrd = new NrrdData(true);
	  
	  int axis = 0,  incr = 0;
	  
	  if (mergedata_ == MERGE_LIKE) {
	    axis = 0; // tuple
	    incr = 0; // tuple case.
	    
	    // if all scalars being merged, need to increment axis
	    bool same_size = true;
	    for (int n=0; n<(int)join_me.size()-1; n++) {
	      if (join_me[n]->dim != join_me[n+1]->dim) 
		same_size = false;
	    }
	    if (same_size)
	      incr = 1; // tuple case.
	  } else if (mergedata_ == 2) {
	    axis = join_me[0]->dim; // time
	    incr = 1;               // time
	  }
	  
	  onrrd->nrrd = nrrdNew();
	  if (nrrdJoin(onrrd->nrrd, &join_me[0], join_me.size(), axis, incr)) {
	    char *err = biffGetDone(NRRD);
	    error(string("Join Error: ") +  err);
	    free(err);
	    error_ = true;
	    return;
	  }
	  
	  // set new kinds for joined nrrds
	  if (join_me.size() == 3)
	    onrrd->nrrd->axis[0].kind = nrrdKind3Vector;
	  else if (join_me.size() == 6)
	    onrrd->nrrd->axis[0].kind = nrrdKind3DSymTensor;
	  else 
	    onrrd->nrrd->axis[0].kind = nrrdKindDomain;
	  for(int i=1; i<onrrd->nrrd->dim; i++) 
	    onrrd->nrrd->axis[i].kind = nrrdKindDomain;
	  
	  
	  if (mergedata_ == MERGE_TIME) {
	    onrrd->nrrd->axis[axis].label = "Time";
	    // remove all numbers from name
	    // FIX ME
	    string s(join_me[0]->axis[0].label);
	    new_label.clear();
	    
	    const string nums("0123456789");
	    
	    //cout << "checking in_name " << s << endl;
	    // test against valid char set.
	    
	    // FIX ME
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
	  }
	  
	  // Take care of tuple axis label.
	  // FIX ME
	  onrrd->nrrd->axis[0].label = strdup(new_label.c_str());
	  
	  // Copy the properties.
	  NrrdDataHandle handle = NrrdDataHandle(onrrd);
	  
	  *((PropertyManager *) handle.get_rep()) =
	    *((PropertyManager *) n.get_rep());
	  
	  // clear the nrrds;
	  vec.clear();
	  vec.push_back(handle);
	}
      }
    }
  }

  unsigned int cc = 0;

  for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {

    for( unsigned int jc=0; jc<ids[ic].size(); jc++ )
      ports[ids[ic][jc]] = cc + (nHandles[ic].size() == 1 ? 0 : jc);

    for( unsigned int jc=0; jc<nHandles[ic].size(); jc++ ) {
      if( cc < MAX_PORTS )
	nHandles_[cc] = nHandles[ic][jc];

      ++cc;
    }
  }

  char *portStr = scinew char[paths.size()*4 + 2 ];

  portStr[0] = '\0';

  for( unsigned int ic=0; ic<paths.size(); ic++ )
    sprintf( portStr, "%s %3d", portStr, ports[ic] );
      
  if( cache ) {
    // Update the ports in the GUI.
    ostringstream str;
    str << id << " updateSelection {" << portStr << "}";

    gui->execute(str.str().c_str());
  }

  if( cc > MAX_PORTS )
    warning( "More data than availible ports." );

  for( unsigned int ic=cc; ic<MAX_PORTS; ic++ )
    nHandles_[ic] = NULL;


  for( unsigned int ic=0; ic<MAX_PORTS; ic++ ) {
    // Get a handle to the output double port.
    if( nHandles_[ic].get_rep() ) {

      char portNumber[4];
      sprintf( portNumber, "%d", ic );

      string portName = string("Output ") +
	string(portNumber) +
	string( " Nrrd" );
      
      NrrdOPort *ofield_port = (NrrdOPort *) get_oport(portName);
    
      if (!ofield_port) {
	error("Unable to initialize "+name+"'s " + portName + " oport\n");
	return;
      }

      ofield_port->set_cache( cache );

      // Send the data downstream
      if( last )
	ofield_port->send( nHandles_[ic] );
      else
	ofield_port->send_intermediate( nHandles_[ic] );
    }
  }


  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Selected Index");

  if (!omatrix_port) {
    error("Unable to initialize oport 'Selected Index'.");
    return;
  }

  ColumnMatrix *selected = scinew ColumnMatrix(1);
  selected->put(0, 0, (double)which);

  bool isDependent = (dependence_.get()==string("dependent"));

  if ( last )
    omatrix_port->send(MatrixHandle(selected));
  else if ( isDependent )
    omatrix_port->send(MatrixHandle(selected));
  else
    omatrix_port->send_intermediate(MatrixHandle(selected));
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

    // Remove the first space.
    path.erase( 1, 1 );

    open = 0;

    // Replace all of spaces that are not in braces '{}'
    // with a forward slash '/'.
    for( unsigned int i=0; i<path.length(); i++ ) {

      if( path[i] == '{' ) {
	open++;
      } else if( path[i] == '}' ) {
	open--;
      } else if( !open && path[i] == ' ' ) {
	path[i] = '/';
      }
    }

    // If still open there is a brace mismatch
    if( open ) {
      error( "Found a path with mismatched braces - " + path );
      return;
    } else {
      std::string::size_type last;

      // Remove the braces.
      while( (last = path.find( "{" )) != std::string::npos )
	path.erase( last, 1 );
    
      while( (last = path.find( "}" )) != std::string::npos )
	path.erase( last, 1 );

      // Get the dataset name which is after the last forward slash '/'.
      last = path.find_last_of( "/" );
      
      // Get the dataset name.
      string dataset( path.substr( last+1, path.length()-last) );
      
      // Remove the dataset name from the path.
      path.erase( last, path.length()-last);

      
      paths.push_back( path );
      datasets.push_back( dataset );
    }
  }
}


unsigned int
HDF5DataReader::parseAnimateDatasets( vector<string>& paths,
				      vector<string>& datasets,
				      vector< vector<string> >& frame_paths,
				      vector< vector<string> >& frame_datasets )
{
  frame_paths.clear();
  frame_datasets.clear();

  unsigned int i, j;

  if( paths.size() == 1 ) {
    frame_paths.push_back( paths );
    frame_datasets.push_back( datasets );
    return 1;
  }

  string comp = paths[0];

  for( i=1; i<paths.size(); i++ ) {
    unsigned int len = paths[i].length();

    // Reduce the size of the comparison to the smallest string.
    if( comp.length() > len )
      comp.replace( len, comp.length()- len, "" );    
    
    // Mark the characters that are different.
    for( unsigned int c=0; c<comp.length(); c++ ) {
      if( comp[c] != paths[i][c] )
	comp[c] = '?';
    }
  }

  unsigned int d1, d2;

  for( d1=0; d1<comp.length(); d1++ )
    if( comp[d1] == '?' )
      break;

  for( d2=d1; d2<comp.length(); d2++ )
    if( comp[d2] != '?' )
      break;

  string root = paths[0].substr( 0, d1 );

  vector <string> times;

  // Get all of the times.
  for( i=0; i<paths.size(); i++ ) {
    string time = paths[i].substr( d1, d2-d1 );

    for( j=0; j<times.size(); j++ ) {
      if( time == times[j] )
	break;
    }

    if( j==times.size() )
      times.push_back( time );
  }

  std::sort( times.begin(), times.end() );

  bool warn = 0;
    
  // Sort the datasets by time.
  for( j=0; j<times.size(); j++ ) {

    string base = root + times[j];

    vector< string > path_list;
    vector< string > dataset_list;

    for( i=0; i<paths.size(); i++ ) {
      if( paths[i].find( base ) != string::npos ) {

	path_list.push_back( paths[i] );
	dataset_list.push_back( datasets[i] );
      }
    }

    frame_paths.push_back( path_list );
    frame_datasets.push_back( dataset_list );

    // Make sure the paths are the same.
    if( j>0 ) {

      if( frame_paths[0].size()    != frame_paths[j].size() ||
	  frame_datasets[0].size() != frame_datasets[j].size() ) {
	warning( "Animation path and or dataset size mismatch" );
	warn = 1;
      } else {

	for( i=0; i<frame_paths[0].size(); i++ ) {

	  if( frame_paths[0][i].substr(d2, frame_paths[0][i].length()-d2) !=
	      frame_paths[j][i].substr(d2, frame_paths[0][i].length()-d2) ||
	      frame_datasets[0][i] != frame_datasets[j][i] ) {
	    warning( "Animation path and or dataset mismatch" );
	    warning( frame_paths[0][i] + " " + frame_datasets[0][i] );
	    warning( frame_paths[j][i] + " " + frame_datasets[j][i] );
	    warn = 1;
	    break;
	  }
	}
      }
    }
  }

  if( warn ) {
    ostringstream str;
    str << times.size() << " frames found ";
    str << " but there was a mismatch.";
    warning( str.str() );
  }

  return times.size();
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
    error( "Error opening file - " + filename);
    return idims;
  }

  /* Open the group in the file. */
  if( (g_id = H5Gopen(file_id, group.c_str())) < 0 ) {
    error( "Error opening group - " + group );
    return idims;
  }

  /* Open the dataset in the file. */
  if( (ds_id = H5Dopen(g_id, dataset.c_str())) < 0 ) {
    error( "Error opening dataset - " + dataset );
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

#ifdef HAVE_HDF5
herr_t add_attribute(hid_t group_id, const char * aname, void* op_data) {
  herr_t status;

  hid_t attr_id = H5Aopen_name(group_id, aname);

  if (attr_id < 0) {
    cerr << "Unable to open attribute \"" << aname << "\"" << endl;
    status = -1;
  } else {

    hid_t type_id = H5Aget_type( attr_id );
    hid_t file_space_id = H5Aget_space( attr_id );

    if( file_space_id < 0 ) {
      cerr << "Unable to open data " << endl;
      return -1;
    }
    
    hid_t mem_type_id;

    switch (H5Tget_class(type_id)) {
    case H5T_STRING:
      // String
      if(H5Tis_variable_str(type_id)) {                    
	mem_type_id = H5Tcopy(H5T_C_S1);                        
	H5Tset_size(mem_type_id, H5T_VARIABLE);                 
      } else {                                      
	mem_type_id = H5Tcopy(type_id);
	H5Tset_cset(mem_type_id, H5T_CSET_ASCII);
      }

      break;

    case H5T_INTEGER:
      // Integer
      mem_type_id = H5T_NATIVE_INT;
      break;

    case H5T_FLOAT:
      if (H5Tequal(type_id, H5T_IEEE_F32BE) ||
	  H5Tequal(type_id, H5T_IEEE_F32LE) ||
	  H5Tequal(type_id, H5T_NATIVE_FLOAT)) {
	// Float
	mem_type_id = H5T_NATIVE_FLOAT;

      } else if (H5Tequal(type_id, H5T_IEEE_F64BE) ||
		 H5Tequal(type_id, H5T_IEEE_F64LE) ||
		 H5Tequal(type_id, H5T_NATIVE_DOUBLE) ||
		 H5Tequal(type_id, H5T_NATIVE_LDOUBLE)) {
	// Double
	mem_type_id = H5T_NATIVE_DOUBLE;

      } else {
	cerr << "Undefined HDF5 float" << endl;
	return -1;
      }
      break;
    default:
      cerr << "Unknown or unsupported HDF5 data type" << endl;
      return -1;
    }
    
    /* Get the rank (number of dims) in the space. */
    int ndims = H5Sget_simple_extent_ndims(file_space_id);

    hsize_t *dims = new hsize_t[ndims];

    /* Get the dims in the space. */
    int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

    if( ndim != ndims ) {
      cerr << "Data dimensions not match." << endl;
      return -1;
    }


    int cc = 1;

    for( int ic=0; ic<ndims; ic++ )
      cc *= dims[ic];

    int size;

    if( H5Tget_size(type_id) > H5Tget_size(mem_type_id) )
      size = cc * H5Tget_size(type_id);
    else
      size = cc * H5Tget_size(mem_type_id);

    char *data = new char[size+1];

    if( data == NULL ) {
      cerr << "Can not allocate enough memory for the data" << endl;
      return -1;
    }

    status = H5Aread(attr_id, mem_type_id, data);

    if( status < 0 ) {
      cerr << "Can not read data" << endl;
      return -1;
    }

    // This ensures that the last character is NULL.
    ((char*) data)[size] = '\0';

    NrrdData * nrrd = (NrrdData *) (op_data);


    if( cc == 1 ) {
      if (mem_type_id == H5T_NATIVE_INT)
	nrrd->set_property( aname, ((int*) data)[0], false );
      else if (mem_type_id == H5T_NATIVE_FLOAT)
	nrrd->set_property( aname, ((float*) data)[0], false );
      else if (mem_type_id == H5T_NATIVE_DOUBLE)
	nrrd->set_property( aname, ((double*) data)[0], false );
      else if( H5Tget_class(type_id) == H5T_STRING ) {
	if(H5Tis_variable_str(type_id))
	  nrrd->set_property( aname, ((char*) data)[0], false );
	else
	  nrrd->set_property( aname, string((char*) data), false );
      }
    } else {
      ostringstream str;

      for( int ic=0; ic<cc; ic++ ) {
	if (mem_type_id == H5T_NATIVE_INT)
	  str << ((int*) data)[ic];
	else if (mem_type_id == H5T_NATIVE_FLOAT)
	  str << ((float*) data)[ic];
	else if (mem_type_id == H5T_NATIVE_DOUBLE)
	  str << ((double*) data)[ic];
	else if( H5Tget_class(type_id) == H5T_STRING ) {
	  if(H5Tis_variable_str(type_id))
	    str << ((char*) data)[ic];
	  else
	    str << "\"" << (char*) data  << "\"";
	}

	if( ic<cc-1)
	  str << ", ";
      }

      nrrd->set_property( aname, str.str(), false );
    }

    H5Tclose(type_id);

    H5Aclose(attr_id);

    delete data;
  }    

  return status;
}
#endif


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
    error( "Error opening file - " + filename);
    return NULL;
  }

  /* Open the group in the file. */
  if( (g_id = H5Gopen(file_id, group.c_str())) < 0 ) {
    error( "Error opening group - " + group);
    return NULL;
  }

  /* Open the dataset in the file. */
  if( (ds_id = H5Dopen(g_id, dataset.c_str())) < 0 ) {
    error( "Error opening data space - " + dataset);
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
    error("Compound HDF5 data types can not be converted into Nrrds.");
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

  hsize_t *count = 0;
  hsize_t *dims = 0;

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

  // FIX ME !!!!!!
  string tuple_type_str(":Scalar");
  NrrdData *nout = scinew NrrdData(true);


  switch(ndims) {
  case 1: 
    nrrdWrap(nout->nrrd, data,
	     nrrd_type, ndims, (unsigned int) count[0]);
    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode);
    nout->nrrd->axis[0].kind = nrrdKindDomain;
    break;

  case 2: 
    {
      // If the user asks us to assume vector or tensor data, the
      // assumption is based on the size of the last dimension of the hdf5 data
      // amd will be in the first dimension of the nrrd
      int sz_last_dim = 1;
      if (assumesvt_) { sz_last_dim = count[1];} 
      
      switch (sz_last_dim) {
      case 3: // Vector data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		 (unsigned int) count[0], (unsigned int) count[1]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode);
	tuple_type_str = ":Vector";
	nout->nrrd->axis[0].kind = nrrdKind3Vector;
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	nout->nrrd->axis[0].kind = nrrdKind3DSymTensor;
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 
		 (unsigned int) count[0], (unsigned int) count[1]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode);
	nout->nrrd->axis[0].kind = nrrdKindDomain;
	break;
      };
      nout->nrrd->axis[1].kind = nrrdKindDomain;
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
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode, nrrdCenterNode);
	tuple_type_str = ":Vector";
	nout->nrrd->axis[0].kind = nrrdKind3Vector;
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode, nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	nout->nrrd->axis[0].kind = nrrdKind3DSymTensor;
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims,  
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode, nrrdCenterNode);
	nout->nrrd->axis[0].kind = nrrdKindDomain;
	break;
      };
      nout->nrrd->axis[1].kind = nrrdKindDomain;
      nout->nrrd->axis[2].kind = nrrdKindDomain;
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
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode, nrrdCenterNode, nrrdCenterNode,
			nrrdCenterNode);
	tuple_type_str = ":Vector";
	nout->nrrd->axis[0].kind = nrrdKind3Vector;
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode, nrrdCenterNode, nrrdCenterNode,
			nrrdCenterNode);       
	tuple_type_str = ":Tensor";
	nout->nrrd->axis[0].kind = nrrdKind3DSymTensor;
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims,  
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			nrrdCenterNode, nrrdCenterNode, nrrdCenterNode,
			nrrdCenterNode);
	nout->nrrd->axis[0].kind = nrrdKindDomain;
	break;
      };
      nout->nrrd->axis[1].kind = nrrdKindDomain;
      nout->nrrd->axis[2].kind = nrrdKindDomain;
      nout->nrrd->axis[3].kind = nrrdKindDomain;
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
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode);
	tuple_type_str = ":Vector";
	nout->nrrd->axis[0].kind = nrrdKind3Vector;
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	nout->nrrd->axis[0].kind = nrrdKind3DSymTensor;
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims,  
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode);
	nout->nrrd->axis[0].kind = nrrdKindDomain;
	break;
      };
      nout->nrrd->axis[1].kind = nrrdKindDomain;
      nout->nrrd->axis[2].kind = nrrdKindDomain;
      nout->nrrd->axis[3].kind = nrrdKindDomain;
      nout->nrrd->axis[4].kind = nrrdKindDomain;
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
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);
	tuple_type_str = ":Vector";
	nout->nrrd->axis[0].kind = nrrdKind3Vector;
	break;
	
      case 6: // Tensor data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4], dims[5]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);	
	tuple_type_str = ":Tensor";
	nout->nrrd->axis[0].kind = nrrdKind3DSymTensor;
	break;

      default: // treat the rest as Scalar data
	nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 
		 (unsigned int) count[0], (unsigned int) count[1], 
		 (unsigned int) count[2], (unsigned int) count[3], 
		 (unsigned int) count[4], dims[5]);
	nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);
	nout->nrrd->axis[0].kind = nrrdKindDomain;
	break;
      };
      nout->nrrd->axis[1].kind = nrrdKindDomain;
      nout->nrrd->axis[2].kind = nrrdKindDomain;
      nout->nrrd->axis[3].kind = nrrdKindDomain;
      nout->nrrd->axis[4].kind = nrrdKindDomain;
      nout->nrrd->axis[5].kind = nrrdKindDomain;
    }
    break;
  }

  // FIX ME
  string sink_label = group + "-" + dataset + tuple_type_str;

  // Remove all of the tcl special characters.
  std::string::size_type pos;
  while( (pos = sink_label.find("/")) != string::npos )
    sink_label.replace( pos, 1, "-" );
  while( (pos = sink_label.find("[")) != string::npos )
    sink_label.replace( pos, 1, "_" );
  while( (pos = sink_label.find("]")) != string::npos )
    sink_label.erase( pos, 1 );
  while( (pos = sink_label.find(" ")) != string::npos )
    sink_label.replace( pos, 1, "_" );

  nout->nrrd->axis[0].label = strdup(sink_label.c_str());

  for( int ic=0; ic<ndims; ic++ ) {
    char tmpstr[16];

    sprintf( tmpstr, "%d", (int) (count[ic]) );
    nout->nrrd->axis[ic+1].label = strdup(tmpstr);
  }

  delete dims;
  delete count;

  std::string parent = group;

  while( parent.length() > 0 ) {
  
    hid_t p_id = H5Gopen(file_id, parent.c_str());

    /* Open the group in the file. */
    if( p_id < 0 ) {
      error( "Error opening group. " );
    } else {
      H5Aiterate(p_id, NULL, add_attribute, nout);

      /* Terminate access to the group. */ 
      if( (status = H5Gclose(p_id)) < 0 )
	error( "Can not close file space." );
    }

    // Remove the last group name from the path.
    std::string::size_type last = parent.find_last_of("/");
    parent.erase( last, parent.length()-last);
  }

  parent = "/";

  hid_t p_id = H5Gopen(file_id, parent.c_str());
  
  /* Open the group in the file. */
  if( p_id < 0 ) {
    error( "Error opening group. " );
  } else {
    H5Aiterate(p_id, NULL, add_attribute, nout);
    
    /* Terminate access to the group. */ 
    if( (status = H5Gclose(p_id)) < 0 )
      error( "Can not close file space." );
  }


  /* Terminate access to the data space. */ 
  if( (status = H5Sclose(file_space_id)) < 0 )
    error( "Can not close file space." );

  /* Terminate access to the dataset. */
  if( (status = H5Dclose(ds_id)) < 0 )
    error( "Can not close file space." );

  /* Terminate access to the group. */ 
  if( (status = H5Gclose(g_id)) < 0 )
    error( "Can not close file space." );

  /* Terminate access to the file. */ 
  if( (status = H5Fclose(file_id)) < 0 )
    error( "Can not close file space." );



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
  if (args[1] == "play" || args[1] == "stop" || args[1] == "step") {
    execmode_ = string(args[1]);

  } else if (args[1] == "update_file") {
#ifdef HAVE_HDF5
    filename_.reset();
    string new_filename(filename_.get());

    if( new_filename.length() == 0 )
      return;

    // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
    struct stat64 buf;
    if (stat64(new_filename.c_str(), &buf)) {
#else
    struct stat buf;
    if (stat(new_filename.c_str(), &buf)) {
#endif
      error( string("Updating - File not found ") + new_filename );
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

      char* tmpdir = getenv( "SCIRUN_TMP_DIR" );

      string tmp_filename;

      if( tmpdir )
	tmp_filename = tmpdir + string( "/" );
      else
	tmp_filename = string( "/tmp/" );

      tmp_filename.append( new_filename, last, new_filename.length()-last );
      tmp_filename.append( ".dump" );

      std::ofstream sPtr( tmp_filename.c_str() );

      if( !sPtr ) {
	error( string("Unable to open output file: ") + tmp_filename );
	return;
      }
  
      if( DataIO::HDF5Dump_file( new_filename.c_str(), &sPtr ) < 0 ) {
	error( string("Could not create dump file: ") + tmp_filename );
	return;
      }

      sPtr.flush();
      sPtr.close();

#ifdef HAVE_STAT64
    if (stat64(tmp_filename.c_str(), &buf)) {
#else
    if (stat(tmp_filename.c_str(), &buf)) {
#endif
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

    if( new_filename.length() == 0 || new_datasets.length() == 0 )
      return;

    // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
    struct stat64 buf;
    if (stat64(new_filename.c_str(), &buf)) {
#else
    struct stat buf;
    if (stat(new_filename.c_str(), &buf)) {
#endif
      error( string("Selection - File not found ") + new_filename );
      return;
    }

    // If we haven't read yet, or if it's a new filename, 
    //  or if the datestamp has changed -- then read...
#ifdef __sgi
    time_t new_filemodification = buf.st_mtim.tv_sec;
#else
    time_t new_filemodification = buf.st_mtime;
#endif

    vector<string> paths;
    vector<string> datasets;
    
    parseDatasets( new_datasets, paths, datasets );

    if( new_filename         != tmp_filename_ || 
	new_filemodification != tmp_filemodification_ ||
	new_datasets         != tmp_datasets_ ) {

      tmp_filemodification_ = new_filemodification;
      tmp_filename_         = new_filename;
      tmp_datasets_         = new_datasets;

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

      nDims_.reset();
      bool set = (ndims != (unsigned int) nDims_.get());

      if( !set ) {
	for( unsigned int ic=0; ic<ndims; ic++ ) {
	  gDims_[ic]->reset();
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

    animate_.reset();

    // If animate is one figure out how many frames we have.
    if( animate_.get() == 1 ) {
      vector< vector<string> > frame_paths;
      vector< vector<string> > frame_datasets;

      int nframes =
	parseAnimateDatasets( paths, datasets, frame_paths, frame_datasets);

      if( nframes != selectable_max_.get() ) {
	ostringstream str;
	str << id << " update_animate_range 0 " << nframes-1;
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


int
HDF5DataReader::increment(int which, int lower, int upper)
{
  // Do nothing if no range.
  if (upper == lower) {
    if (playmode_.get() == "once")
      execmode_ = string("stop");
    return upper;
  }

  const int inc_amount = Max(1, Min(upper, inc_amount_.get()));
  which += inc_ * inc_amount;

  if (which > upper) {
    if (playmode_.get() == "bounce1") {
      inc_ *= -1;
      return increment(upper, lower, upper);
    } else if (playmode_.get() == "bounce2") {
      inc_ *= -1;
      return upper;
    } else {
      if (playmode_.get() == "once")
	execmode_ = string("stop");
      return lower;
    }
  }

  if (which < lower) {
    if (playmode_.get() == "bounce1") {
      inc_ *= -1;
      return increment(lower, lower, upper);
    } else if (playmode_.get() == "bounce2") {
      inc_ *= -1;
      return lower;
    } else {
      if (playmode_.get() == "once")
	execmode_ = string("stop");
      return upper;
    }
  }
  return which;
}

void
HDF5DataReader::animate_execute( string new_filename,
				 vector< vector<string> >& frame_paths,
				 vector< vector<string> >& frame_datasets )
{
  update_state(NeedData);

  reset_vars();

  // Update the increment.
  const int start = range_min_.get();
  const int end = range_max_.get();
  if (playmode_.get() == "once" || playmode_.get() == "loop")
    inc_ = (start>end)?-1:1;

  // If the current value is invalid, reset it to the start.
  int lower = start;
  int upper = end;
  if (lower > upper) {int tmp = lower; lower = upper; upper = tmp; }
  if (current_.get() < lower || current_.get() > upper) {
    current_.set(start);
    inc_ = (start>end)?-1:1;
  }

  // Cash execmode and reset it in case we bail out early.

  int which = current_.get();

  bool cache = (playmode_.get() != "inc_w_exec");

  if (execmode_ == "step") {

    which = increment(which, lower, upper);
    current_.set(which);
    current_.reset();

    ReadandSendData( new_filename, frame_paths[which],
		     frame_datasets[which], true, cache, which );

  } else if (execmode_ == "play") {

    if (playmode_.get() == "once" && which >= end)
      which = start;

    bool stop;
    do {
      int delay = delay_.get();

      int next = increment(which, lower, upper);
      stop = (execmode_ == "stop");

      current_.set(which);
      current_.reset();

      ReadandSendData( new_filename, frame_paths[which],
		       frame_datasets[which], stop, stop ? cache : true,
		       which );

      if (!stop && delay > 0) {
	const unsigned int secs = delay / 1000;
	const unsigned int msecs = delay % 1000;
	if (secs)  { sleep(secs); }
	if (msecs) { usleep(msecs * 1000); }
      }

      if (playmode_.get() == "once" || !stop )
	which = next;

    } while (!stop);
  } else {
    
    ReadandSendData( new_filename, frame_paths[which],
		     frame_datasets[which], true, cache, which );

    if (playmode_.get() == "inc_w_exec") {
      which = increment(which, lower, upper);
      current_.set(which);
      current_.reset();
    }
  }

  execmode_ = string("none");
}

} // End namespace DataIO

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
#include <sci_defs/hdf5_defs.h>
#include <sci_defs/stat64_defs.h>

#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/ColumnMatrix.h>

#include <Packages/DataIO/Dataflow/Modules/Readers/HDF5DataReader.h>

#include <sys/stat.h>
#include <fstream>
#include <algorithm>

#ifdef HAVE_HDF5
#include "hdf5.h"
#include "HDF5Dump.h"
#endif

namespace DataIO {

using namespace SCIRun;

DECLARE_MAKER(HDF5DataReader)

HDF5DataReader::HDF5DataReader(GuiContext *context)
  : Module("HDF5DataReader", context, Source, "Readers", "DataIO"),
    power_app_(context->subVar("power_app")),
    power_app_cmd_(context->subVar("power_app_commmand")),

    filename_(context->subVar("filename")),
    datasets_(context->subVar("datasets")),
    dumpname_(context->subVar("dumpname")),
    ports_(context->subVar("ports")),

    nDims_(context->subVar("ndims")),

    mergeData_(context->subVar("mergeData")),
    assumeSVT_(context->subVar("assumeSVT")),
    animate_(context->subVar("animate")),

    animate_frame_(ctx->subVar("animate_frame")),
    animate_tab_(ctx->subVar("animate_tab")),
    basic_tab_(ctx->subVar("basic_tab")),
    extended_tab_(ctx->subVar("extended_tab")),
    playmode_tab_(ctx->subVar("playmode_tab")),

    selectable_min_(ctx->subVar("selectable_min")),
    selectable_max_(ctx->subVar("selectable_max")),
    selectable_inc_(ctx->subVar("selectable_inc")),
    range_min_(ctx->subVar("range_min")),
    range_max_(ctx->subVar("range_max")),
    playmode_(ctx->subVar("playmode")),
    current_(ctx->subVar("current")),
    execmode_(ctx->subVar("execmode")),
    delay_(ctx->subVar("delay")),
    inc_amount_(ctx->subVar("inc-amount")),
    update_type_(ctx->subVar("update_type")),
    inc_(1),

    mergedata_(-1),
    assumesvt_(-1),

    update_(false),
    which_(-1),

    loop_(false),
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
  std::string::size_type pos;

  string nrrdName1, nrrdName2, grp1, grp2;

  h1->get_property( "Name", nrrdName1 );
  h2->get_property( "Name", nrrdName2 );

  if (mergedata_ == MERGE_LIKE) {
    grp1 = nrrdName1;
    grp2 = nrrdName2;

    pos = grp1.find_last_of("-");
    if( pos != std::string::npos )
      grp1.erase( pos, grp1.length()-pos );

    pos = grp2.find_last_of("-");
    if( pos != std::string::npos )
      grp2.erase( pos, grp2.length()-pos );

    if( grp1 != grp2 )
      return false;
  }

  // The names are the only properties that are allowed to be different
  // when merging so remove them before testing the rest of the properties.
  h1->remove_property( "Name" );
  h2->remove_property( "Name" );
  
  bool pass = true;
  
  if( (*((PropertyManager *) h1.get_rep()) !=
       *((PropertyManager *) h2.get_rep())) ) pass = false;
  
  // Restore the names
  h1->set_property( "Name", nrrdName1, false );
  h2->set_property( "Name", nrrdName2, false );
  
  if( !pass )
    return false;

  Nrrd* n1 = h1->nrrd; 
  Nrrd* n2 = h2->nrrd;
    
  if (n1->type != n2->type)
    return false;

  if (n1->dim  != n2->dim)
    return false;
  
  // Compare the dimensions.
  for (int i=0; i<n1->dim; i++) {
    if (n1->axis[i].size != n2->axis[i].size)
      return false;
  }

  return true;
}


void HDF5DataReader::execute() {

  update_state(NeedData);

#ifdef HAVE_HDF5
  filename_.reset();
  datasets_.reset();

  string filename(filename_.get());
  string datasets(datasets_.get());

  if( filename.length() == 0 ) {
    error( string("No HDF5 file selected.") );
    return;
  }
  
  if( datasets.length() == 0 ) {
    error( string("No HDF5 datasets selected.") );
    return;
  }

  // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
  struct stat64 buf;
  if (stat64(filename.c_str(), &buf) == -1)
#else
  struct stat buf;
  if (stat(filename.c_str(), &buf) == -1)
#endif
  {
    error( string("HDF5 File not found ") + filename );
    return;
  }

#ifdef __sgi
  time_t filemodification = buf.st_mtim.tv_sec;
#else
  time_t filemodification = buf.st_mtime;
#endif

  update_ = false;

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
  if( filename         != old_filename_ || 
      filemodification != old_filemodification_ || 
      datasets         != old_datasets_ ) {

    old_filemodification_ = filemodification;
    old_filename_         = filename;
    old_datasets_         = datasets;

    sel_filemodification_ = filemodification;
    sel_filename_         = filename;
    sel_datasets_         = datasets;

    update_ = true;
  }

  update_state(JustStarted);

  bool resend = false;

  // get all the actual values from gui.
  reset_vars();

  if( mergedata_ != mergeData_.get() ||
      assumesvt_ != assumeSVT_.get() ) {

    mergedata_ = mergeData_.get();
    assumesvt_ = assumeSVT_.get();

    update_ = true;
  }

  for( int ic=0; ic<MAX_DIMS; ic++ ) {

    if( starts_ [ic] != gStarts_ [ic]->get() ||
	counts_ [ic] != gCounts_ [ic]->get() ||
	strides_[ic] != gStrides_[ic]->get() ) {

      starts_ [ic] = gStarts_ [ic]->get();
      counts_ [ic] = gCounts_ [ic]->get();
      strides_[ic] = gStrides_[ic]->get();
    
      update_ = true;
    }

    if( dims_[ic] != gDims_[ic]->get() ) {
      dims_[ic] = gDims_[ic]->get();
      update_ = true;
    }

    if( starts_[ic] + (counts_[ic]-1) * strides_[ic] >= dims_[ic] ) {
    
      error( "Data selection exceeds bounds." );
      error( "Decrease the start or count or increase the stride." );
      return;
    }
  }

  vector< string > pathList;
  vector< string > datasetList;
  
  parseDatasets( datasets, pathList, datasetList );

  if( animate_.get() ) {

    vector< vector<string> > frame_paths;
    vector< vector<string> > frame_datasets;
    
    unsigned int nframes =
      parseAnimateDatasets( pathList, datasetList, frame_paths, frame_datasets);
    

    // If there is a current index matrix, use it.
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Current Index");

    if (!imatrix_port) {
      error("Unable to initialize iport 'Current Index'.");
      return;
    }

    MatrixHandle mHandle;
    if (imatrix_port->get(mHandle) && mHandle.get_rep()) {
      int which = (int) (mHandle->get(0, 0));

      if( 0 <= which && which <= (int) frame_paths.size() ) {
	if( error_ ||
	    update_ ||
	    current_.get() != which ) {
	  current_.set(which);
	  current_.reset();
	  
	  ReadandSendData( filename, frame_paths[which],
			   frame_datasets[which], true, which );
	} else
	  resend = true;
      } else {

	ostringstream str;
	str << "Input index is out of range ";
	str << "0 <= " << which << " <= " << (int) frame_paths.size();

	error( str.str() );

	return;
      }
    } else {
      if( nframes-1 != selectable_max_.get() ) {
	selectable_max_.set(nframes-1);
	selectable_max_.reset();
      }

      resend = animate_execute( filename, frame_paths, frame_datasets );
    }
  } else if( error_ ||
	     update_ ){
    error_ = false;

    ReadandSendData( filename, pathList, datasetList, true, -1 );

  } else
    resend = true;


  if( resend ) {
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

	ofield_port->send( nHandles_[ic] );
      }
    }

    // Get a handle to the output double port.
    if( mHandle_.get_rep() ) {
      MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Selected Index");

      if (!omatrix_port) {
	error("Unable to initialize oport 'Selected Index'.");
	return;
      }

      omatrix_port->send(mHandle_);
    }
  }

  update_state(Completed);

#else  
  error( "No HDF5 availible." );
  
#endif
}

void HDF5DataReader::ReadandSendData( string& filename,
				      vector< string >& pathList,
				      vector< string >& datasetList,
				      bool cache,
				      int which ) {

  vector< vector<NrrdDataHandle> > nHandles;
  vector< vector<int> > ids;

  vector< int > ports;
    
  for( unsigned int ic=0; ic<pathList.size(); ic++ ) {
    ports.push_back( -1 );

    NrrdDataHandle nHandle =
      readDataset(filename, pathList[ic], datasetList[ic]);

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
      error( "No handle for - " + pathList[ic] + "/" + datasetList[ic] );
      return;
    }
  }

  // Merge the like datatypes together.
  if( mergedata_ ) {

    vector<vector<NrrdDataHandle> >::iterator iter = nHandles.begin();
    while (iter != nHandles.end()) {

      vector<NrrdDataHandle> &vec = *iter;
      ++iter;

      if( vec.size() > 1) {

	if( assumesvt_ && vec.size() != 3 && vec.size() != 6 && vec.size() != 9) {
	  warning( "Assuming Vector and Matrix data but can not merge into a Vector or Matrix because there are not 3, 6, or 9 nrrds that are alike." );
	  continue;
	}
	  
	vector<Nrrd*> join_me;
	vector<NrrdDataHandle>::iterator niter = vec.begin();

	NrrdDataHandle n = *niter;
	++niter;
	join_me.push_back(n->nrrd);

	string nrrdName, groupName, dataName;

	n->get_property( "Name", groupName );
	std::string::size_type pos = groupName.find_last_of("-");
	if( pos != std::string::npos )
	  groupName.erase( pos, groupName.length()-pos );

	n->get_property( "Name", dataName );
	pos = dataName.find_last_of("-"); // Erase the Group
	if( pos != std::string::npos )
	  dataName.erase( 0, pos );
	pos = dataName.find_last_of(":"); // Erase the Kind
	if( pos != std::string::npos )
	  dataName.erase( pos, dataName.length()-pos );

	nrrdName = groupName + dataName;

	while (niter != vec.end()) {
	  NrrdDataHandle n = *niter;
	  ++niter;
	  join_me.push_back(n->nrrd);

	  if (mergedata_ == MERGE_LIKE) {
	    n->get_property( "Name", dataName );
	    pos = dataName.find_last_of("-"); // Erase the Group
	    if( pos != std::string::npos )
	      dataName.erase( 0, pos );
	    pos = dataName.find_last_of(":"); // Erase the Kind
	    if( pos != std::string::npos )
	      dataName.erase( pos, dataName.length()-pos );

	    nrrdName += dataName;
	  }	  
	}

	NrrdData* onrrd = new NrrdData();
	  
	int axis = 0; // axis
	int incr = 1; // incr.
	
	onrrd->nrrd = nrrdNew();
	if (nrrdJoin(onrrd->nrrd, &join_me[0], join_me.size(), axis, incr)) {
	  char *err = biffGetDone(NRRD);
	  error(string("Join Error: ") +  err);
	  free(err);
	  error_ = true;
	  return;
	}

	// set new kinds for joined nrrds
	if (assumesvt_ && join_me.size() == 3) {
	  onrrd->nrrd->axis[0].kind = nrrdKind3Vector;
	  nrrdName += string(":Vector");
	} else if (assumesvt_ && join_me.size() == 6) {
	  onrrd->nrrd->axis[0].kind = nrrdKind3DSymMatrix;
	  nrrdName += string(":Matrix");
	} else if (assumesvt_ && join_me.size() == 9) {
	  onrrd->nrrd->axis[0].kind = nrrdKind3DMatrix;
	  nrrdName += string(":Matrix");
	} else {
	  onrrd->nrrd->axis[0].kind = nrrdKindDomain;
	  nrrdName += string(":Scalar");
	}

	for(int i=1; i<onrrd->nrrd->dim; i++) {
	  onrrd->nrrd->axis[i].kind = nrrdKindDomain;
	  onrrd->nrrd->axis[i].label = join_me[0]->axis[i].label;
	}

	if (mergedata_ == MERGE_LIKE) {
	  onrrd->nrrd->axis[axis].label = strdup("Merged Data");
	} else if (mergedata_ == MERGE_TIME) {
	  onrrd->nrrd->axis[axis].label = "Time";

	  // remove all numbers from name
	  string s(nrrdName);
	  nrrdName.clear();
	    
	  const string nums("0123456789");
	    
	  // test against valid char set.	    
	  for(string::size_type i = 0; i < s.size(); i++) {
	    bool in_set = false;
	    for (unsigned int c = 0; c < nums.size(); c++) {
	      if (s[i] == nums[c]) {
		in_set = true;
		break;
	      }
	    }
	      
	    if (in_set) { nrrdName.push_back('X' ); }
	    else        { nrrdName.push_back(s[i]); }	      
	  }
	}

	// Copy the properties.
	NrrdDataHandle handle = NrrdDataHandle(onrrd);
	  
	*((PropertyManager *) handle.get_rep()) =
	  *((PropertyManager *) n.get_rep());
	  
	// Take care of the axis label and the nrrd name.
	onrrd->set_property( "Name", nrrdName, false );

	// clear the nrrds;
	vec.clear();
	vec.push_back(handle);
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

  char *portStr = scinew char[pathList.size()*4 + 2 ];

  portStr[0] = '\0';

  for( unsigned int ic=0; ic<pathList.size(); ic++ )
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

      nHandles_[ic]->set_property("Source",string("HDF5"), false);

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

      // Send the data downstream
      ofield_port->set_cache( cache );
      ofield_port->send( nHandles_[ic] );
    }
  }

  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Selected Index");

  if (!omatrix_port) {
    error("Unable to initialize oport 'Selected Index'.");
    return;
  }

  ColumnMatrix *selected = scinew ColumnMatrix(1);
  selected->put(0, 0, (double)which);

  mHandle_ = MatrixHandle(selected);
  omatrix_port->send( mHandle_ );
}


void HDF5DataReader::parseDatasets( string datasets,
				    vector<string>& pathList,
				    vector<string>& datasetList )
{
  int open = 0;

  std::string::size_type cc = 0;
  std::string::size_type bb = 0;
  std::string::size_type ee = 0;

  string path, dataset;

  while( cc < datasets.length() ) {

    bb = ee = cc;

    if( datasets[bb] == '{' ) {
      // Multiple datasets.

      open = 1;
      ee = bb + 1;

      while( open && ee < datasets.length() ) {
	ee++;

	if( datasets[ee] == '{' )
	  open++;
	else if( datasets[ee] == '}' )
	  open--;
      }

      path = datasets.substr( bb+1, ee-bb-1);

      cc = ee + 2;

    } else {
      // Single Dataset
      path = datasets;

      cc = datasets.length();
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
      std::string::size_type pos;

      // Remove the braces.
      while( (pos = path.find( "{" )) != std::string::npos )
	path.erase( pos, 1 );
    
      while( (pos = path.find( "}" )) != std::string::npos )
	path.erase( pos, 1 );

      // Get the dataset name which is after the pos forward slash '/'.
      pos = path.find_last_of( "/" );
      
      // Get the dataset name.
      string dataset( path.substr( pos+1, path.length()-pos) );
      
      // Remove the dataset name from the path.
      path.erase( pos, path.length()-pos);

      // Just incase the dataset is at the root.
      if( path.length() == 0 ) path = string( "/" );

      pathList.push_back( path );
      datasetList.push_back( dataset );
    }
  }
}


unsigned int
HDF5DataReader::parseAnimateDatasets( vector<string>& pathList,
				      vector<string>& datasetList,
				      vector< vector<string> >& frame_paths,
				      vector< vector<string> >& frame_datasets )
{
  frame_paths.clear();
  frame_datasets.clear();

  unsigned int i, j;

  if( pathList.size() == 1 ) {
    frame_paths.push_back( pathList );
    frame_datasets.push_back( datasetList );
    return 1;
  }

  string comp = pathList[0];

  for( i=1; i<pathList.size(); i++ ) {
    unsigned int len = pathList[i].length();

    // Reduce the size of the comparison to the smallest string.
    if( comp.length() > len )
      comp.replace( len, comp.length()- len, "" );    
    
    // Mark the characters that are different.
    for( unsigned int c=0; c<comp.length(); c++ ) {
      if( comp[c] != pathList[i][c] )
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

  string root = pathList[0].substr( 0, d1 );

  vector <string> times;

  // Get all of the times.
  for( i=0; i<pathList.size(); i++ ) {
    string time = pathList[i].substr( d1, d2-d1 );

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

    for( i=0; i<pathList.size(); i++ ) {
      if( pathList[i].find( base ) != string::npos ) {

	path_list.push_back( pathList[i] );
	dataset_list.push_back( datasetList[i] );
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
  herr_t status = 0;

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

static string HDF5Attribute_error_msg;

herr_t add_attribute(hid_t group_id, const char * aname, void* op_data) {
  herr_t status = 0;

  hid_t attr_id = H5Aopen_name(group_id, aname);

  if (attr_id < 0) {
    HDF5Attribute_error_msg =
      string("Unable to open attribute \"") + aname + "\"";
    status = -1;
  } else {

    hid_t type_id = H5Aget_type( attr_id );
    hid_t file_space_id = H5Aget_space( attr_id );

    if( file_space_id < 0 ) {
      HDF5Attribute_error_msg = "Unable to open data ";
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
	HDF5Attribute_error_msg = "Undefined HDF5 float";
	return -1;
      }
      break;
    case H5T_REFERENCE:
	return status;
      break;
    default:
      HDF5Attribute_error_msg = "Unknown or unsupported HDF5 data type";
      return -1;
    }
    
    /* Get the rank (number of dims) in the space. */
    int ndims = H5Sget_simple_extent_ndims(file_space_id);

    hsize_t *dims = new hsize_t[ndims];

    /* Get the dims in the space. */
    int ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);

    if( ndim != ndims ) {
      HDF5Attribute_error_msg = "Data dimensions not match.";
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
      HDF5Attribute_error_msg = "Can not allocate enough memory for the data";
      return -1;
    }

    status = H5Aread(attr_id, mem_type_id, data);

    if( status < 0 ) {
      HDF5Attribute_error_msg = "Can not read data";
      return status;
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

  herr_t status = 0;
 
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
    error("At this time HDF5 Compound types can not be converted into Nrrds.");
    error_ = true;
    return NULL;
    break;

  case H5T_REFERENCE:
    error("At this time HDF5 Reference types are not followed.");
    error("Please select the actual object.");
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

#if H5_VERS_RELEASE == 2
      hssize_t *start = new hssize_t[ndims];
#else
      hsize_t *start  = new hsize_t[ndims];
#endif
      hsize_t *stride = new hsize_t[ndims];
      hsize_t *block  = new hsize_t[ndims];

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

  // Stuff the data into the NRRD.
  NrrdData *nout = scinew NrrdData();

  // If the user asks us to assume vector or matrix data, the
  // assumption is based on the size of the last dimension of the hdf5 data
  // amd will be in the first dimension of the nrrd
  int sz_last_dim = 1;
  if (assumesvt_)
    sz_last_dim = dims[ndims-1];

  // The nrrd ordering is opposite of HDF5 ordering so swap the dimensions
  for(int i=0; i<ndims/2; i++ ) {
    int swap = count[i];
    count[i] = count[ndims-1-i];
    count[ndims-1-i] = swap;
  }

  switch(ndims) {
  case 1: 
    nrrdWrap(nout->nrrd, data,
	     nrrd_type, ndims, (unsigned int) count[0]);
    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode);
    break;
      
  case 2: 
    nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 
	     (unsigned int) count[0], (unsigned int) count[1]);
    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode);
    break;
      
  case 3: 
    nrrdWrap(nout->nrrd, data, nrrd_type, ndims,  
	     (unsigned int) count[0], (unsigned int) count[1], 
	     (unsigned int) count[2]);
    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);
    break;
      
  case 4: 
    nrrdWrap(nout->nrrd, data, nrrd_type, ndims,  
	     (unsigned int) count[0], (unsigned int) count[1], 
	     (unsigned int) count[2], (unsigned int) count[3]);
    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode,
		    nrrdCenterNode);
    break;
      
  case 5: 
    nrrdWrap(nout->nrrd, data, nrrd_type, ndims,  
	     (unsigned int) count[0], (unsigned int) count[1], 
	     (unsigned int) count[2], (unsigned int) count[3], 
	     (unsigned int) count[4]);
    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode);
    break;
      
  case 6: 
    nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 
	     (unsigned int) count[0], (unsigned int) count[1], 
	     (unsigned int) count[2], (unsigned int) count[3], 
	     (unsigned int) count[4], count[5]);
    nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
		    nrrdCenterNode, nrrdCenterNode);
    break;
  }
   
  string nrrdName = group + "-" + dataset;

  // Remove all of the tcl special characters.
  std::string::size_type pos;
  while( (pos = nrrdName.find(":")) != string::npos )
    nrrdName.replace( pos, 1, "-" );
  while( (pos = nrrdName.find("/")) != string::npos )
    nrrdName.replace( pos, 1, "-" );
  while( (pos = nrrdName.find("[")) != string::npos )
    nrrdName.replace( pos, 1, "_" );
  while( (pos = nrrdName.find("]")) != string::npos )
    nrrdName.erase( pos, 1 );
  while( (pos = nrrdName.find(" ")) != string::npos )
    nrrdName.replace( pos, 1, "_" );


  switch (sz_last_dim) {
  case 3: // Vector data
    nrrdName += ":Vector";
    nout->nrrd->axis[0].kind = nrrdKind3Vector;
    break;
	  
  case 6: // Matrix data
    nrrdName += ":Matrix";
    nout->nrrd->axis[0].kind = nrrdKind3DSymMatrix;
    break;
	  
  case 9: // Matrix data
    nrrdName += ":Matrix";
    nout->nrrd->axis[0].kind = nrrdKind3DMatrix;
    break;
	  
  default: // treat the rest as Scalar data
    nrrdName += ":Scalar";
    nout->nrrd->axis[0].kind = nrrdKindDomain;
    break;
  };

  for( int i=1; i<ndims; i++ )
    nout->nrrd->axis[i].kind = nrrdKindDomain;


  nout->set_property( "Name", nrrdName, false );

  delete dims;
  delete count;

  // Add the attributs from the dataset.
  if( H5Aiterate(ds_id, NULL, add_attribute, nout) < 0 ) {
    error( HDF5Attribute_error_msg );
  }

  std::string parent = group;

  // Add the attributs from the parents.
  while( parent.length() > 0 ) {
  
    hid_t p_id = H5Gopen(file_id, parent.c_str());

    /* Open the group in the file. */
    if( p_id < 0 ) {
      error( "Error opening group. " );
    } else if( H5Aiterate(p_id, NULL, add_attribute, nout) < 0 ) {
      error( HDF5Attribute_error_msg );
    } else {

      /* Terminate access to the group. */ 
      if( (status = H5Gclose(p_id)) < 0 )
	error( "Can not close file space." );
    }

    // Remove the last group name from the path.
    std::string::size_type pos = parent.find_last_of("/");
    parent.erase( pos, parent.length()-pos);
  }

  // Add the attributs from the top level.
  parent = "/";

  hid_t p_id = H5Gopen(file_id, parent.c_str());
  
  /* Open the group in the file. */
  if( p_id < 0 ) {
    error( "Error opening group. " );
  } else if( H5Aiterate(p_id, NULL, add_attribute, nout) < 0 ) {
    error( HDF5Attribute_error_msg );
  } else {
    
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

  if (args[1] == "check_dumpfile") {
#ifdef HAVE_HDF5
    filename_.reset();

    string filename(filename_.get());
  
    if( filename.length() == 0 )
      return;

    // Dump file name change
    string dumpname = getDumpFileName( filename );

    if( dumpname != dumpname_.get() ) {
      dumpname_.set( dumpname );
      dumpname_.reset();
    }

    // Dump file not available or out of date .
    if( checkDumpFile( filename, dumpname ) ) {
      createDumpFile( filename, dumpname );
    }

#else
    error( "No HDF5 availible." );
#endif

  } else if (args[1] == "update_file") {
#ifdef HAVE_HDF5

    bool update = false;

    filename_.reset();
    string new_filename(filename_.get());

    if( new_filename.length() == 0 ) {
      error( string("No HDF5 file.") );
      return;
    }

    // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
    struct stat64 buf;
    if (stat64(new_filename.c_str(), &buf) == -1) {
#else
    struct stat buf;
    if (stat(new_filename.c_str(), &buf) == -1) {
#endif
      error( string("Update - File not found ") + new_filename );
      return;
    }

    // If we haven't read yet, or if it's a new filename, 
    //  or if the datestamp has changed -- then read...
#ifdef __sgi
    time_t new_filemodification = buf.st_mtim.tv_sec;
#else
    time_t new_filemodification = buf.st_mtime;
#endif

    // Dump file name change
    string new_dumpname = getDumpFileName( new_filename );

    if( new_filename         != sel_filename_ || 
	new_filemodification != sel_filemodification_) {

      sel_filename_         = new_filename;
      sel_filemodification_ = new_filemodification;

      update = true;

    } else {

      update = checkDumpFile( new_filename, new_dumpname );
    }

    if( update ) {
      createDumpFile( new_filename, new_dumpname  );
    
      // Update the treeview in the GUI.
      ostringstream str;
      str << id << " build_tree " << new_dumpname;
      
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

    if( new_datasets.length() == 0 )
      return;

    vector<string> pathList;
    vector<string> datasetList;
    
    parseDatasets( new_datasets, pathList, datasetList );

    if( new_datasets != sel_datasets_ ) {
      sel_datasets_ = new_datasets;

      unsigned long ndims = 0;

      for( int ic=0; ic<MAX_DIMS; ic++ )
	dims_[ic] = 1;

      for( unsigned int ic=0; ic<pathList.size(); ic++ ) {

	vector<int> dims =
	  getDatasetDims( new_filename, pathList[ic], datasetList[ic] );

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
	parseAnimateDatasets( pathList, datasetList, frame_paths, frame_datasets);

      if( nframes-1 != selectable_max_.get() ) {
	selectable_max_.set(nframes-1);
	selectable_max_.reset();
      }
    }
#else
    error( "No HDF5 availible." );
#endif

  } else {
    Module::tcl_command(args, userdata);
  }
}

    
string
HDF5DataReader::getDumpFileName( string filename ) {

  string dumpname;

  std::string::size_type pos = filename.find_last_of( "/" );

  if( pos == string::npos )
    pos = 0;
  else
    pos++;

  char* tmpdir = getenv( "SCIRUN_TMP_DIR" );

  if( tmpdir )
    dumpname = tmpdir + string( "/" );
  else
    dumpname = string( "/tmp/" );

  dumpname.append( filename, pos, filename.length()-pos );
  dumpname.append( ".dump" );

  return dumpname;
}

bool
HDF5DataReader::checkDumpFile( string filename, string dumpname ) {

  bool recreate = false;

  // Read the status of this file so we can compare modification timestamps
#ifdef HAVE_STAT64
  struct stat64 buf;
  if (stat64(filename.c_str(), &buf) == -1)
#else
  struct stat buf;
  if (stat(filename.c_str(), &buf) == -1)
#endif
  {
    error( string("HDF5 File not found ") + filename );
    return false;
  }

#ifdef __sgi
  time_t filemodification = buf.st_mtim.tv_sec;
  time_t dumpfilemodification = 0;
#else
  time_t filemodification = buf.st_mtime;
  time_t dumpfilemodification = 0;
#endif

  // Read the status of this dumpfile so we can compare modification timestamps
#ifdef HAVE_STAT64
  if (stat64(dumpname.c_str(), &buf) == -1)
#else
  if (stat(dumpname.c_str(), &buf) == -1)
#endif
  {
    warning( string("HDF5 Dump File not found ") + dumpname +
	     " ... recreating.");
    recreate = true;
  } else {

#ifdef __sgi
    dumpfilemodification = buf.st_mtim.tv_sec;
#else
    dumpfilemodification = buf.st_mtime;
#endif
    if( dumpfilemodification < filemodification ) {
      warning( string("HDF5 Dump File is out of date ") + dumpname +
	       " ... recreating.");
      recreate = true;
    }
  }

  return recreate;
}


int
HDF5DataReader::createDumpFile( string filename, string dumpname ) {

  std::ofstream sPtr( dumpname.c_str() );

  if( !sPtr ) {
    error( string("Unable to open output file: ") + dumpname );
    gui->execute( "reset_cursor" );
    return -1;
  }

  HDF5Dump hdf( &sPtr );

  if( hdf.file( filename ) < 0 ) {
    error( hdf.error() );
    gui->execute( "reset_cursor" );

    sPtr.flush();
    sPtr.close();

    return -1;
  }

  sPtr.flush();
  sPtr.close();

  return 0;
}


int
HDF5DataReader::increment(int which, int lower, int upper)
{
  // Do nothing if no range.
  if (upper == lower) {
    if (playmode_.get() == "once")
      execmode_.set( "stop" );
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
	execmode_.set( "stop" );
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
	execmode_.set( "stop" );
      return upper;
    }
  }

  return which;
}

bool
HDF5DataReader::animate_execute( string new_filename,
				 vector< vector<string> >& frame_paths,
				 vector< vector<string> >& frame_datasets )
{
  bool resend = false;

  update_state(NeedData);

  reset_vars();

  // Cache var
  bool cache = (playmode_.get() != "inc_w_exec");

  // Get the current start and end.
  const int start = range_min_.get();
  const int end   = range_max_.get();

  int lower = start;
  int upper = end;
  if (lower > upper) {int tmp = lower; lower = upper; upper = tmp; }

  // Update the increment.
  if (playmode_.get() == "once" || playmode_.get() == "loop")
    inc_ = (start>end)?-1:1;

  // If the current value is invalid, reset it to the start.
  if (current_.get() < lower || upper < current_.get()) {
    current_.set(start);
    inc_ = (start>end)?-1:1;
  }

  // Cache execmode and reset it in case we bail out early.
  const string execmode = execmode_.get();
  
  int which = current_.get();

  // If updating, we're done for now.
  if (execmode == "update") {

  } else if (execmode == "step") {
    which = increment(current_.get(), lower, upper);

    ReadandSendData( new_filename, frame_paths[which],
		     frame_datasets[which], cache, which );

  } else if (execmode == "stepb") {
    inc_ *= -1;
    which = increment(current_.get(), lower, upper);
    inc_ *= -1;

    ReadandSendData( new_filename, frame_paths[which],
		     frame_datasets[which], cache, which );

  } else if (execmode == "play") {

    if( !loop_ ) {
      if (playmode_.get() == "once" && which >= end)
	which = start;
    }

    ReadandSendData( new_filename, frame_paths[which],
		     frame_datasets[which], cache, which );
    
    // User may have changed the execmode to stop so recheck.
    execmode_.reset();
    if ( loop_ = (execmode_.get() == "play") ) {
      const int delay = delay_.get();
      
      if( delay > 0) {
	const unsigned int secs = delay / 1000;
	const unsigned int msecs = delay % 1000;
	if (secs)  { sleep(secs); }
	if (msecs) { usleep(msecs * 1000); }
      }
    
      int next = increment(which, lower, upper);    

      // Incrementing may cause a stop in the execmode so recheck.
      execmode_.reset();
      if( loop_ = (execmode_.get() == "play") ) {
	which = next;

	want_to_execute();
      }
    }

  } else {

    if( execmode == "rewind" )
      which = start;

    else if( execmode == "fforward" )
      which = end;

    if( update_ ||
	which != which_ ) {
      ReadandSendData( new_filename, frame_paths[which],
		       frame_datasets[which], cache, which );
    }
    else
      resend = true;

    if (playmode_.get() == "inc_w_exec") {
      which = increment(which, lower, upper);
    }
  }

  which_ = which;

  current_.set(which);

  return resend;
}

} // End namespace DataIO

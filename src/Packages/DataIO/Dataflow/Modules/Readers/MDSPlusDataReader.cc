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
 *  MDSPlusDataReader.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <sci_defs.h>

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>

#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>

#include <Packages/DataIO/share/share.h>
#include <Packages/DataIO/Core/ThirdParty/mdsPlusReader.h>

namespace DataIO {

using namespace SCIRun;
using namespace SCITeem;

#define MAX_PORTS 8

class DataIOSHARE MDSPlusDataReader : public Module {
public:
  MDSPlusDataReader(GuiContext *context);

  virtual ~MDSPlusDataReader();

  bool is_mergeable(NrrdDataHandle h1, NrrdDataHandle h2) const;

  virtual void execute();
  virtual void tcl_command(GuiArgs& args, void* userdata);

  NrrdDataHandle readDataset( string& server,
			      string& tree,
			      int&    shot,
			      string& signal );

protected:

  GuiInt    nEntries_;
  GuiString sServer_;
  GuiString sTree_;
  GuiString sShot_;

  GuiString searchServer_;
  GuiString searchTree_;
  GuiString searchShot_;
  GuiString searchSignal_;
  GuiInt    searchRegexp_;

  GuiInt mergeData_;
  GuiInt assumeSVT_;

  vector< GuiString* > gServer_;
  vector< GuiString* > gTree_;
  vector< GuiString* > gShot_;
  vector< GuiString* > gSignal_;
  vector< GuiString* > gStatus_;
  vector< GuiString* > gPort_;

  unsigned int entries_;
  string server_;
  string tree_;
  string shot_;

  int mergedata_;
  int assumesvt_;

  vector< string > servers_;
  vector< string > trees_;
  vector< int    > shots_;
  vector< string > signals_;
  vector< string > status_;
  vector< unsigned int > ports_;

  bool error_;

  NrrdDataHandle nHandles_[MAX_PORTS];
};


DECLARE_MAKER(MDSPlusDataReader)


MDSPlusDataReader::MDSPlusDataReader(GuiContext *context)
  : Module("MDSPlusDataReader", context, Source, "Readers", "DataIO"),
    nEntries_(context->subVar("num-entries")),
    sServer_(context->subVar("server")),
    sTree_(context->subVar("tree")),
    sShot_(context->subVar("shot")),
    
    searchServer_(context->subVar("search-server")),
    searchTree_(context->subVar("search-tree")),
    searchShot_(context->subVar("search-shot")),
    searchSignal_(context->subVar("search-signal")),
    searchRegexp_(context->subVar("search-regexp")),

    mergeData_(context->subVar("mergeData")),
    assumeSVT_(context->subVar("assumeSVT")),
    entries_(0),
    error_(-1)
{
}

MDSPlusDataReader::~MDSPlusDataReader(){
}

// Allows nrrds to join along tuple, scalar and vector sets can not be joined,
// or allows for a multiple identical nrrds to assume a time series, 
// and be joined along a new time axis. 
bool
MDSPlusDataReader::is_mergeable(NrrdDataHandle h1, NrrdDataHandle h2) const
{
  if( mergedata_ != 2 )
    if( (*((PropertyManager *) h1.get_rep()) !=
	 *((PropertyManager *) h2.get_rep())) ) return false;

  if( h1->concat_tuple_types() != h2->concat_tuple_types() ) return false;

  Nrrd* n1 = h1->nrrd;
  Nrrd* n2 = h2->nrrd;

  if (n1->type != n2->type) return false;
  if (n1->dim  != n2->dim)  return false;
  for (int i=1; i<n1->dim; i++)
    if (n1->axis[i].size != n2->axis[i].size) return false;

  return true;
}


void MDSPlusDataReader::execute(){

#ifdef HAVE_MDSPLUS

  // Save off the defaults
  entries_ = nEntries_.get();                  // Number of entries
  server_ = sServer_.get();                // MDS+ Default Server
  tree_ = sTree_.get();                    // MDS+ Default Tree 
  shot_ = atoi( sShot_.get().c_str() );  // MDS+ Default shot

  int entries = gServer_.size();

  // Remove the old entries that are not needed.
  for( int ic=entries-1; ic>=(int)entries_; ic-- ) {
    delete( gServer_[ic] );
    delete( gTree_[ic] );
    delete( gShot_[ic] );
    delete( gSignal_[ic] );
    delete( gStatus_[ic] );
    delete( gPort_[ic] );

    gServer_.pop_back();
    gTree_.pop_back();
    gShot_.pop_back();
    gSignal_.pop_back();
    gStatus_.pop_back();
    gPort_.pop_back();

    servers_.pop_back();
    trees_.pop_back();
    shots_.pop_back();
    signals_.pop_back();
    status_.pop_back();
    ports_.pop_back();
  }

  // Add new entries that are needed.
  for( unsigned int ic=entries; ic<entries_; ic++ ) {
    char idx[24];

    sprintf( idx, "server-%d", ic );
    gServer_.push_back(new GuiString(ctx->subVar(idx)) );
    sprintf( idx, "tree-%d", ic );
    gTree_.push_back(new GuiString(ctx->subVar(idx)) );
    sprintf( idx, "shot-%d", ic );
    gShot_.push_back(new GuiString(ctx->subVar(idx)) );
    sprintf( idx, "signal-%d", ic );
    gSignal_.push_back(new GuiString(ctx->subVar(idx)) );
    sprintf( idx, "status-%d", ic );
    gStatus_.push_back(new GuiString(ctx->subVar(idx)) );
    sprintf( idx, "port-%d", ic );
    gPort_.push_back(new GuiString(ctx->subVar(idx)) );

    servers_.push_back("");
    trees_.push_back("");
    shots_.push_back(-1);
    signals_.push_back("");
    status_.push_back("Unkown");
    ports_.push_back(99);
  }

  bool update = false;

  for( unsigned int ic=0; ic<entries_; ic++ ) {
    gServer_[ic]->reset();
    gTree_[ic]->reset();
    gShot_[ic]->reset();
    gSignal_[ic]->reset();

    gStatus_[ic]->set( string( "Unknown" ) );
    gPort_[ic]->set( string( "na" ) );

    string tmpStr;

    tmpStr = gServer_[ic]->get();
    if( tmpStr != servers_[ic] ) {
      servers_[ic] = tmpStr;
      update = true;
    }

    tmpStr = gTree_[ic]->get();
    if( tmpStr != trees_[ic] ) {
      trees_[ic] = tmpStr;
      update = true;
    }

    tmpStr = gShot_[ic]->get();
    if( atoi( tmpStr.c_str() ) != shots_[ic] ) {
      shots_[ic] = atoi( tmpStr.c_str() );
      update = true;
    }

    tmpStr = gSignal_[ic]->get();
    if( tmpStr != servers_[ic] ) {
      signals_[ic] = tmpStr;
      update = true;
    }
  }

  if( update == true ||

      error_ == true ||

      mergedata_ != mergeData_.get() ||
      assumesvt_ != assumeSVT_.get() ) {

    error_ = false;

    mergedata_ = mergeData_.get();
    assumesvt_ = assumeSVT_.get();

    vector< vector<NrrdDataHandle> > nHandles;
    vector< vector<int> > ids;
    
    for( unsigned int ic=0; ic<entries_; ic++ ) {

      gStatus_[ic]->set( string( "Reading" ) );

      NrrdDataHandle nHandle =
	readDataset( servers_[ic], trees_[ic], shots_[ic], signals_[ic] );

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

	if (! inserted) {
	  vector<NrrdDataHandle> nrrdSet;
	  nrrdSet.push_back( nHandle );
	  nHandles.push_back( nrrdSet );

	  vector<int> idSet;
	  idSet.push_back( ic );
	  ids.push_back( idSet );
	}
	gStatus_[ic]->set( string("Okay") );
      } else
	gStatus_[ic]->set( string("Error") );
    }

    // merge the like datatypes together.
    if( mergedata_ ) {

      vector<vector<NrrdDataHandle> >::iterator iter = nHandles.begin();
      while (iter != nHandles.end()) {

	vector<NrrdDataHandle> &vec = *iter;
	++iter;

	if (vec.size() > 1) {
	  
	  NrrdData* onrrd = new NrrdData(true);
	  string new_label("");
	  string axis_label("");

	  // check if this is a time axis merge or a tuple axis merge.
	  vector<Nrrd*> join_me;
	  vector<NrrdDataHandle>::iterator niter = vec.begin();

	  NrrdDataHandle n = *niter;
	  ++niter;
	  join_me.push_back(n->nrrd);
	  new_label = n->nrrd->axis[0].label;

	  while (niter != vec.end()) {
	    NrrdDataHandle n = *niter;
	    ++niter;
	    join_me.push_back(n->nrrd);
	    new_label += string(",") + n->nrrd->axis[0].label;
	  }	  

	  int axis,  incr;

	  if (mergedata_ == 1) {
	    axis = 0; // tuple
	    incr = 0; // tuple case.
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

	  if (mergedata_ == 2) {
	    onrrd->nrrd->axis[axis].label = "Time";
	    // remove all numbers from name
	    string s(join_me[0]->axis[0].label);
	    new_label.clear();

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
	  }
	   
	  // Take care of tuple axis label.
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

    unsigned int cc = 0;

    for( unsigned int ic=0; ic<nHandles.size(); ic++ ) {

      for( unsigned int jc=0; jc<ids[ic].size(); jc++ )
	ports_[ids[ic][jc]] = cc + (nHandles[ic].size() == 1 ? 0 : jc);

      for( unsigned int jc=0; jc<nHandles[ic].size(); jc++ ) {
	if( cc < MAX_PORTS )
	  nHandles_[cc] = nHandles[ic][jc];

	++cc;
      }
    }

    char portStr[6];

    for( unsigned int ic=0; ic<entries_; ic++ ) {
      if( 0 <= ports_[ic] && ports_[ic] <= 7 )
	sprintf( portStr, "%3d", ports_[ic] );
      else
	sprintf( portStr, "na" );

      gPort_[ic]->set( string( portStr ) );
    }

    if( cc > MAX_PORTS )
      warning( "More data than availible ports." );

    for( unsigned int ic=cc; ic<MAX_PORTS; ic++ )
      nHandles_[ic] = NULL;
  } else {
    remark( "Already read data " );
  }

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
#else  
  error( "No HDF5 availible." );
  
#endif
}

   
NrrdDataHandle MDSPlusDataReader::readDataset( string& server,
					       string& tree,
					       int&    shot,
					       string& signal ) {

#ifdef HAVE_MDSPLUS
  MDSPlusReader mds;

  int trys = 0;
  int retVal;

  while( trys < 10 && (retVal = mds.connect( server.c_str()) ) == -2 ) {
    remark( "Waiting for the connection to become free." );
    sleep( 1 );
    trys++;
  }

  /* Connect to MDSplus */
  if( retVal == -2 ) {
    error( "Connection to Mds Server " + server + " too busy ... giving up.");
    error_ = true;
    return NULL;
  }
  else if( retVal < 0 ) {
    error( "Connecting to Mds Server " + server );
    error_ = true;
    return NULL;
  }
  else
    remark( "Conecting to Mds Server " + server );

  // Open tree
  trys = 0;

  while( trys < 10 && (retVal = mds.open( tree.c_str(), shot) ) == -2 ) {
    remark( "Waiting for the tree and shot to become free." );
    sleep( 1 );
    trys++;
  }

  if( retVal == -2 ) {
    ostringstream str;
    str << "Opening " << tree << " tree and shot " << shot << " too busy ... giving up.";
    error_ = true;
    return NULL;
  }
  if( retVal < 0 ) {
    ostringstream str;
    str << "Opening " << tree << " tree and shot " << shot;
    error( str.str() );
    error_ = true;
    return NULL;
  }
  else {
    ostringstream str;
    str << "Opening " << tree << " tree and shot " << shot;
    remark( str.str() );
  }

  if( !mds.valid( signal ) ) {
    ostringstream str;
    str << "Invalid signal " << signal;
    error( str.str() );
    error_ = true;
    return NULL;
  }

  // Get the mds data rank from the signal.
  int* dims;
  int ndims = mds.dims( signal, &dims );

  if( ndims < 1 ) {
    ostringstream str;
    str << "Zero dimension signal " << signal;
    error( str.str() );
    error_ = true;
    return NULL;
  }

  // Get the mds data type from the signal
  int mds_data_type = mds.type( signal );

  unsigned int dtype, nrrd_type;

  switch (mds_data_type) {
  case DTYPE_UCHAR:
  case DTYPE_USHORT:
  case DTYPE_ULONG:
  case DTYPE_ULONGLONG:
  case DTYPE_CHAR:
  case DTYPE_SHORT:
  case DTYPE_LONG:
  case DTYPE_LONGLONG:
    dtype = DTYPE_LONG;
    nrrd_type = get_nrrd_type<int>();
    break;

  case DTYPE_FLOAT:
  case DTYPE_FS:
    dtype = DTYPE_FLOAT;
    nrrd_type = get_nrrd_type<float>();
    break;

  case DTYPE_DOUBLE:
  case DTYPE_FT:
    dtype = DTYPE_DOUBLE;
    nrrd_type = get_nrrd_type<double>();
    break;
  case DTYPE_CSTRING:
    {
      ostringstream str;
      str << "String data is not supported for signal " << signal;
      error( str.str() );
      error_ = true;
    }
  default:
    {
      ostringstream str;
      str << "Unknown type (" << mds_data_type << ") for signal " << signal;
      error( str.str() );
      error_ = true;
    }
    
    return NULL;
  }

  // Get the total size of the signal.
  int size = 1;
  for( int i=0; i<ndims; i++ )
    size *= dims[i];

  // Get the mds data from the signal
  void* data = mds.values( signal, dtype );

  mds.disconnect();

  //  Load the fields with the mesh and data.
  if( data != NULL && dims[0] ) {

    // Stuff the data into the NRRD.
    string tuple_type_str(":Scalar");
    int sink_size = 1;
    NrrdData *nout = scinew NrrdData(false);

    switch(ndims) {
    case 1:
      nrrdWrap(nout->nrrd, data,
	       nrrd_type, ndims+1, sink_size, (unsigned int) dims[0]);
      nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
		      nrrdCenterNode);
      break;
    case 2: 
      {
	// If the user asks us to assume vector or tensor data, the
	// assumption is based on the size of the last dimension.
	int sz_last_dim = 1;
	if (assumesvt_) { sz_last_dim = dims[1];} 
      
	switch (sz_last_dim) {
	case 3: // Vector data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		   (unsigned int) dims[0], (unsigned int) dims[1]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);
	  tuple_type_str = ":Vector";
	  break;
	
	case 6: // Tensor data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		   (unsigned int) dims[0], (unsigned int) dims[1]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);	
	  tuple_type_str = ":Tensor";
	  break;

	default: // treat the rest as Scalar data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		   (unsigned int) dims[0], (unsigned int) dims[1]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);	
	  break;
	};
      }

    case 3: 
      {
	// If the user asks us to assume vector or tensor data, the
	// assumption is based on the size of the last dimension.
	int sz_last_dim = 1;
	if (assumesvt_) { sz_last_dim = dims[2];} 
      
	switch (sz_last_dim) {
	case 3: // Vector data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	  tuple_type_str = ":Vector";
	  break;
	
	case 6: // Tensor data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);	
	  tuple_type_str = ":Tensor";
	  break;

	default: // treat the rest as Scalar data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
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
	if (assumesvt_) { sz_last_dim = dims[3]; } 

	switch (sz_last_dim) {
	case 3: // Vector data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode,
			  nrrdCenterNode);
	  tuple_type_str = ":Vector";
	  break;
	
	case 6: // Tensor data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode,
			  nrrdCenterNode);
	
	  tuple_type_str = ":Tensor";
	  break;

	default: // treat the rest as Scalar data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
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
	if (assumesvt_) { sz_last_dim = dims[4];} 
      
	switch (sz_last_dim) {
	case 3: // Vector data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3], 
		   (unsigned int) dims[4]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);
	  tuple_type_str = ":Vector";
	  break;
	
	case 6: // Tensor data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3], 
		   (unsigned int) dims[4]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);	
	  tuple_type_str = ":Tensor";
	  break;

	default: // treat the rest as Scalar data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3], 
		   (unsigned int) dims[4]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
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
	if (assumesvt_) { sz_last_dim = dims[5];} 
      
	switch (sz_last_dim) {
	case 3: // Vector data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 3, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3], 
		   (unsigned int) dims[4], dims[5]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	  tuple_type_str = ":Vector";
	  break;
	
	case 6: // Tensor data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims, 6, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3], 
		   (unsigned int) dims[4], dims[5]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);	
	  tuple_type_str = ":Tensor";
	  break;

	default: // treat the rest as Scalar data
	  nrrdWrap(nout->nrrd, data, nrrd_type, ndims+1, sink_size, 
		   (unsigned int) dims[0], (unsigned int) dims[1], 
		   (unsigned int) dims[2], (unsigned int) dims[3], 
		   (unsigned int) dims[4], dims[5]);
	  nrrdAxisInfoSet(nout->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	  break;
	};
      }
      break;
    }

    string sink_label;
    sink_label.clear();

    // Remove the MDS characters that are illegal in Nrrds.
    const string nums("\\:.");
	      
    // test against valid char set.
    for(string::size_type i = 0; i < signal.size(); i++) {
      bool in_set = false;
      for (unsigned int c = 0; c < nums.size(); c++) {
	if (signal[i] == nums[c]) {
	  in_set = true;
	  break;
	}
      }
      
      if (in_set) { sink_label.push_back('_' ); }
      else        { sink_label.push_back(signal[i]); }
      
    }

    sink_label += tuple_type_str;

    nout->nrrd->axis[0].label = strdup(sink_label.c_str());

    for( int ic=0; ic<ndims; ic++ ) {
      char tmpstr[16];

      sprintf( tmpstr, "%d", (int) (dims[ic]) );
      nout->nrrd->axis[ic+1].label = strdup(tmpstr);
    }

    {
      ostringstream str;
      str << "Read " << signal << " data with ";

      for( int ic=0; ic<ndims; ic++ )
	str << dims[ic] << "  "; 

      str << "elements";

      remark( str.str() );
    }

    free( dims );

    return NrrdDataHandle(nout);
  } else
    return NULL;

#else

  error( "No MDS PLUS availible." );

#endif
}

void MDSPlusDataReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("HDF5DataReader needs a minor command");
    return;
  }

  if (args[1] == "search") {
#ifdef HAVE_HDF5

    searchServer_.reset();
    searchTree_.reset();
    searchShot_.reset();
    searchSignal_.reset();
    searchRegexp_.reset();

    // Get the search strings;
    string server( searchServer_.get() );
    string tree  ( searchTree_.get() );
    int    shot  ( atoi( searchShot_.get().c_str() ) );
    string signal( searchSignal_.get() );
    int    regexp( searchRegexp_.get() );

    MDSPlusReader mds;

    int trys = 0;
    int retVal;

    while( trys < 10 && (retVal = mds.connect( server.c_str()) ) == -2 ) {
      remark( "Waiting for the connection to become free." );
      sleep( 1 );
      trys++;
    }

    /* Connect to MDSplus */
    if( retVal == -2 ) {
      error( "Connection to Mds Server " + server + " too busy ... giving up.");
      error_ = true;
      return;
    }
    else if( retVal < 0 ) {
      error( "Connecting to Mds Server " + server );
      error_ = true;
      return;
    }
    else
      remark( "Conecting to Mds Server " + server );

    // Open tree
    trys = 0;

    while( trys < 10 && (retVal = mds.open( tree.c_str(), shot) ) == -2 ) {
      remark( "Waiting for the tree and shot to become free." );
      sleep( 1 );
      trys++;
    }

    if( retVal == -2 ) {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot << " too busy ... giving up.";
      error_ = true;
      return;
    }
    if( retVal < 0 ) {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      error( str.str() );
      error_ = true;
      return;
    }
    else {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      remark( str.str() );
    }

    vector< string > signals;

    mds.search( signal, regexp, signals );

    for( unsigned int ic=0; ic<signals.size(); ic++ ) {
      // Update the list in the GUI.
      ostringstream str;
      str << id << " setEntry {" << signals[ic] << "}";
      
      gui->execute(str.str().c_str());
    }

#else

    error( "No MDS PLUS availible." );

#endif
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace DataIO

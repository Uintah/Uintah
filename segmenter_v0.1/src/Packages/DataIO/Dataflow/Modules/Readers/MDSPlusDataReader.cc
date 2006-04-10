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

#include <sci_defs/mdsplus_defs.h>

#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>

#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Packages/DataIO/share/share.h>
#include <Packages/DataIO/Core/ThirdParty/mdsPlusReader.h>
#include <Packages/DataIO/Dataflow/Modules/Readers/MDSPlusDump.h>

#include <fstream>

namespace DataIO {

using namespace SCIRun;

#define MAX_PORTS 8
#define MAX_DIMS 6

class DataIOSHARE MDSPlusDataReader : public Module {
protected:
  enum { MERGE_NONE=0,   MERGE_LIKE=1,   MERGE_TIME=2 };

public:
  MDSPlusDataReader(GuiContext *context);

  virtual ~MDSPlusDataReader();

  bool is_mergeable(NrrdDataHandle h1, NrrdDataHandle h2);

  virtual void execute();
  virtual void tcl_command(GuiArgs& args, void* userdata);

  NrrdDataHandle readDataset( string& server,
			      string& tree,
			      int&    shot,
			      string& signal );

protected:

  GuiInt    gui_power_app_;
  GuiString gui_loadServer_;
  GuiString gui_loadTree_;
  GuiString gui_loadShot_;
  GuiString gui_loadSignal_;

  GuiInt    gui_nEntries_;
  GuiString gui_sServer_;
  GuiString gui_sTree_;
  GuiString gui_sShot_;

  GuiString gui_searchServer_;
  GuiString gui_searchTree_;
  GuiString gui_searchShot_;
  GuiString gui_searchSignal_;
  GuiInt    gui_searchPath_;

  GuiInt gui_merge_data_;
  GuiInt gui_assume_svt_;

  vector< GuiString* > gui_Server_;
  vector< GuiString* > gui_Tree_;
  vector< GuiString* > gui_Shot_;
  vector< GuiString* > gui_Signal_;
  vector< GuiString* > gui_Status_;
  vector< GuiString* > gui_Port_;

  vector< string > servers_;
  vector< string > trees_;
  vector< int    > shots_;
  vector< string > signals_;
  vector< string > status_;
  vector< unsigned int > ports_;

  bool execute_error_;

  NrrdDataHandle nrrd_output_handles_[MAX_PORTS];
};


DECLARE_MAKER(MDSPlusDataReader)


MDSPlusDataReader::MDSPlusDataReader(GuiContext *context)
  : Module("MDSPlusDataReader", context, Source, "Readers", "DataIO"),
    gui_power_app_(context->subVar("power_app")),

    gui_loadServer_(context->subVar("load-server")),
    gui_loadTree_(context->subVar("load-tree")),
    gui_loadShot_(context->subVar("load-shot")),
    gui_loadSignal_(context->subVar("load-signal")),

    gui_nEntries_(context->subVar("num-entries")),
    gui_sServer_(context->subVar("server")),
    gui_sTree_(context->subVar("tree")),
    gui_sShot_(context->subVar("shot")),
    
    gui_searchServer_(context->subVar("search-server")),
    gui_searchTree_(context->subVar("search-tree")),
    gui_searchShot_(context->subVar("search-shot")),
    gui_searchSignal_(context->subVar("search-signal")),
    gui_searchPath_(context->subVar("search-path")),

    gui_merge_data_(context->subVar("mergeData")),
    gui_assume_svt_(context->subVar("assumeSVT")),

    execute_error_(-1)
{
  for( unsigned int ic=0; ic<MAX_PORTS; ic++ )
    nrrd_output_handles_[ic] = 0;
}

MDSPlusDataReader::~MDSPlusDataReader(){
}

// Allows nrrds to join along tuple, scalar and vector sets can not be joined,
// or allows for a multiple identical nrrds to assume a time series, 
// and be joined along a new time axis. 
bool
MDSPlusDataReader::is_mergeable(NrrdDataHandle h1, NrrdDataHandle h2)
{
  std::string::size_type pos;

  string nrrdName1, nrrdName2, grp1, grp2;

  h1->get_property( "Name", nrrdName1 );
  h2->get_property( "Name", nrrdName2 );

  if ( gui_merge_data_.get() == MERGE_LIKE) {
    grp1 = nrrdName1;
    grp2 = nrrdName2;
    
    pos = grp1.find_last_of(":");
    if( pos != std::string::npos )
      grp1.erase( pos, grp1.length()-pos ); // Erase the kind
    pos = grp1.find_last_of(":");
    if( pos != std::string::npos )
      grp1.erase( pos, grp1.length()-pos ); // Erase the data name
    
    pos = grp2.find_last_of(":");
    if( pos != std::string::npos )
      grp2.erase( pos, grp2.length()-pos ); // Erase the kind
    pos = grp2.find_last_of(":");
    if( pos != std::string::npos )
      grp2.erase( pos, grp2.length()-pos ); // Erase the data name

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

  Nrrd* n1 = h1->nrrd_; 
  Nrrd* n2 = h2->nrrd_;
    
  if (n1->type != n2->type)
    return false;

  if (n1->dim  != n2->dim)
    return false;
  
  int start = 0;
  if (n1->dim >= n2->dim)
    start = n2->dim;
  else
    start = n1->dim;

  // Compare the dimensions.
  for (unsigned int i=0; i<n1->dim; i++) {
    if (n1->axis[i].size != n2->axis[i].size)
      return false;
  }

  return true;
}


void MDSPlusDataReader::execute(){

#ifdef HAVE_MDSPLUS

  // Save off the defaults
  int entries = gui_Server_.size();

  // Remove the old entries that are not needed.
  for( int ic=entries-1; ic>=gui_nEntries_.get(); ic-- ) {
    delete( gui_Server_[ic] );
    delete( gui_Tree_[ic] );
    delete( gui_Shot_[ic] );
    delete( gui_Signal_[ic] );
    delete( gui_Status_[ic] );
    delete( gui_Port_[ic] );

    gui_Server_.pop_back();
    gui_Tree_.pop_back();
    gui_Shot_.pop_back();
    gui_Signal_.pop_back();
    gui_Status_.pop_back();
    gui_Port_.pop_back();

    servers_.pop_back();
    trees_.pop_back();
    shots_.pop_back();
    signals_.pop_back();
    status_.pop_back();
    ports_.pop_back();
  }

  // Add new entries that are needed.
  for( int ic=entries; ic<gui_nEntries_.get(); ic++ ) {
    char idx[24];

    sprintf( idx, "server-%d", ic );
    gui_Server_.push_back(new GuiString(get_ctx()->subVar(idx)) );
    sprintf( idx, "tree-%d", ic );
    gui_Tree_.push_back(new GuiString(get_ctx()->subVar(idx)) );
    sprintf( idx, "shot-%d", ic );
    gui_Shot_.push_back(new GuiString(get_ctx()->subVar(idx)) );
    sprintf( idx, "signal-%d", ic );
    gui_Signal_.push_back(new GuiString(get_ctx()->subVar(idx)) );
    sprintf( idx, "status-%d", ic );
    gui_Status_.push_back(new GuiString(get_ctx()->subVar(idx)) );
    sprintf( idx, "port-%d", ic );
    gui_Port_.push_back(new GuiString(get_ctx()->subVar(idx)) );

    servers_.push_back("");
    trees_.push_back("");
    shots_.push_back(-1);
    signals_.push_back("");
    status_.push_back("Unkown");
    ports_.push_back(99);
  }

  for( int ic=0; ic<gui_nEntries_.get(); ic++ ) {
    gui_Server_[ic]->reset();
    gui_Tree_[ic]->reset();
    gui_Shot_[ic]->reset();
    gui_Signal_[ic]->reset();

    gui_Status_[ic]->set( string( "Unknown" ) );
    gui_Port_[ic]->set( string( "na" ) );

    string tmpStr;

    tmpStr = gui_Server_[ic]->get();

    if( tmpStr != servers_[ic] ) {
      servers_[ic] = tmpStr;
      inputs_changed_ = true;
    }

    tmpStr = gui_Tree_[ic]->get();
    if( tmpStr != trees_[ic] ) {
      trees_[ic] = tmpStr;
      inputs_changed_ = true;
    }
    
    tmpStr = gui_Shot_[ic]->get();
    if( atoi( tmpStr.c_str() ) != shots_[ic] ) {
      shots_[ic] = atoi( tmpStr.c_str() );
      inputs_changed_ = true;
    }
    
    tmpStr = gui_Signal_[ic]->get();
    if( tmpStr != signals_[ic] ) {
      signals_[ic] = tmpStr;
      inputs_changed_ = true;
    }
  }

  if( inputs_changed_ == true ||

      execute_error_ == true ||

      gui_merge_data_.changed( true ) ||
      gui_assume_svt_.changed( true ) ) {

    update_state( Executing );

    execute_error_ = false;

    vector< vector<NrrdDataHandle> > nHandles;
    vector< vector<int> > ids;
    
    for( int ic=0; ic<gui_nEntries_.get(); ic++ ) {

      gui_Status_[ic]->set( string( "Reading" ) );

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
	gui_Status_[ic]->set( string("Okay") );
      } else
	gui_Status_[ic]->set( string("Error") );
    }

    // merge the like datatypes together.
    if( gui_merge_data_.get() ) {

      vector<vector<NrrdDataHandle> >::iterator iter = nHandles.begin();
      while (iter != nHandles.end()) {

	vector<NrrdDataHandle> &vec = *iter;
	++iter;

	if (vec.size() > 1) {
	  
	  if( gui_assume_svt_.get() && vec.size() != 3 && vec.size() != 6) {
	    warning( "Assuming Vector and Tensor data but can not merge into a Vector or Tensor because there are not 3 or 6 nrrds that are alike." );
	    continue;
	  }
	  
	  vector<Nrrd*> join_me;
	  vector<NrrdDataHandle>::iterator niter = vec.begin();

	  NrrdDataHandle n = *niter;
	  ++niter;
	  join_me.push_back(n->nrrd_);

	  string nrrdName, groupName, dataName;
	  std::string::size_type pos;

	  n->get_property( "Name", groupName );
	  pos = groupName.find_last_of(":"); // Erase the Kind
	  if( pos != std::string::npos )
	    groupName.erase( pos, groupName.length()-pos );

	  pos = groupName.find_last_of(":"); // Erase the Name
	  if( pos != std::string::npos )
	    groupName.erase( pos, groupName.length()-pos );


	  n->get_property( "Name", dataName );
	  pos = dataName.find_last_of(":"); // Erase the Kind
	  if( pos != std::string::npos )
	    dataName.erase( pos, dataName.length()-pos );
	  pos = dataName.find_last_of(":"); // Erase the Group
	  if( pos != std::string::npos )
	    dataName.erase( 0, pos );
	  
	  nrrdName = groupName + dataName;

	  while (niter != vec.end()) {
	    NrrdDataHandle n = *niter;
	    ++niter;
	    join_me.push_back(n->nrrd_);

	    if (gui_merge_data_.get() == MERGE_LIKE) {
	      n->get_property( "Name", dataName );
	      pos = dataName.find_last_of(":"); // Erase the Kind
	      if( pos != std::string::npos )
		dataName.erase( pos, dataName.length()-pos );
	      pos = dataName.find_last_of(":"); // Erase the Group
	      if( pos != std::string::npos )
		dataName.replace( 0, pos, "-" );
	      
	      nrrdName += dataName;
	    }
	  }

	  NrrdData* onrrd = new NrrdData();
	  
	  int axis = 0; // axis
	  int incr = 1; // incr.
	
	  onrrd->nrrd_ = nrrdNew();
	  if (nrrdJoin(onrrd->nrrd_, &join_me[0], join_me.size(), axis, incr)) {
	    char *err = biffGetDone(NRRD);
	    error(string("Join Error: ") +  err);
	    free(err);
	    execute_error_ = true;
	    return;
	  }

	  // set new kinds for joined nrrds
	  if (gui_assume_svt_.get() && join_me.size() == 3) {
	    onrrd->nrrd_->axis[0].kind = nrrdKind3Vector;
	    nrrdName += string(":Vector");
	  } else if (gui_assume_svt_.get() && join_me.size() == 6) {
	    onrrd->nrrd_->axis[0].kind = nrrdKind3DSymMatrix;
	    nrrdName += string(":Matrix");
	  } else if (gui_assume_svt_.get() && join_me.size() == 9) {
	    onrrd->nrrd_->axis[0].kind = nrrdKind3DMatrix;
	    nrrdName += string(":Matrix");
	  } else {
	    onrrd->nrrd_->axis[0].kind = nrrdKindDomain;
	    nrrdName += string(":Scalar");
	  }

	  for(unsigned int i=1; i<onrrd->nrrd_->dim; i++) {
	    onrrd->nrrd_->axis[i].kind = nrrdKindDomain;
	    onrrd->nrrd_->axis[i].label = join_me[0]->axis[i].label;
	  }

	  if (gui_merge_data_.get() == MERGE_LIKE) {
	    onrrd->nrrd_->axis[axis].label = strdup("Merged Data");
	  } else if (gui_merge_data_.get() == MERGE_TIME) {
	    onrrd->nrrd_->axis[axis].label = "Time";

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
	ports_[ids[ic][jc]] = cc + (nHandles[ic].size() == 1 ? 0 : jc);

      for( unsigned int jc=0; jc<nHandles[ic].size(); jc++ ) {
	if( cc < MAX_PORTS ) {
	  nrrd_output_handles_[cc] = nHandles[ic][jc];

	  nrrd_output_handles_[cc]->set_property("Source",string("MDSPlus"), false);
	}

	++cc;
      }
    }

    char portStr[6];

    for( int ic=0; ic<gui_nEntries_.get(); ic++ ) {
      if( 0 <= ports_[ic] && ports_[ic] <= 7 )
	sprintf( portStr, "%3d", ports_[ic] );
      else
	sprintf( portStr, "na" );

      gui_Port_[ic]->set( string( portStr ) );
    }

    if( cc >= MAX_PORTS )
      warning( "More data than availible ports." );

    for( unsigned int ic=cc; ic<MAX_PORTS; ic++ )
      nrrd_output_handles_[ic] = 0;

  } else {
    remark( "Already read data " );
  }

  for( unsigned int ic=0; ic<MAX_PORTS; ic++ ) {
    
    string portName = string("Output ") + to_string(ic) + string( " Nrrd" );
    
    // Send the data downstream
    send_output_handle( portName, nrrd_output_handles_[ic], true );
  }
#else  
  error( "No MDSPlus availible." );
  
#endif
}

   
NrrdDataHandle MDSPlusDataReader::readDataset( string& server,
					       string& tree,
					       int&    shot,
					       string& signal ) {

#ifdef HAVE_MDSPLUS
  MDSPlusReader mds;

  int trys = 0;
  int retVal = 0;

  while( trys < 10 && (retVal = mds.connect( server.c_str()) ) == -2 ) {
    remark( "Waiting for the connection to become free." );
    sleep( 1 );
    trys++;
  }

  /* Connect to MDSplus */
  if( retVal == -2 ) {
    error( "Connection to Mds Server " + server + " too busy ... giving up.");
    execute_error_ = true;
    return NULL;
  }
  else if( retVal < 0 ) {
    error( "Connecting to Mds Server " + server );
    execute_error_ = true;
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
    execute_error_ = true;
    return NULL;
  }
  if( retVal < 0 ) {
    ostringstream str;
    str << "Opening " << tree << " tree and shot " << shot;
    error( str.str() );
    execute_error_ = true;
    return NULL;
  }
  else {
    ostringstream str;
    str << "Opening " << tree << " tree and shot " << shot;
    remark( str.str() );
  }

  if( 0 && !mds.valid( signal ) ) {
    ostringstream str;
    str << "Invalid signal " << signal;
    error( str.str() );
    execute_error_ = true;
    return NULL;
  }

  // Get the mds data rank from the signal.
  unsigned int* dims;
  int ndims = mds.dims( signal, &dims );

  if( ndims < 1 ) {
    ostringstream str;
    str << "Zero dimension signal " << signal;
    error( str.str() );
    execute_error_ = true;
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
      execute_error_ = true;
    }
  default:
    {
      ostringstream str;
      str << "Unknown type (" << mds_data_type << ") for signal " << signal;
      error( str.str() );
      execute_error_ = true;
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
    NrrdData *nout = scinew NrrdData();

    // If the user asks us to assume vector or tensor data, the
    // assumption is based on the size of the last dimension of the hdf5 data
    // amd will be in the first dimension of the nrrd
    int sz_last_dim = 1;
    if (gui_assume_svt_.get())
      sz_last_dim = dims[ndims-1];

    size_t size[NRRD_DIM_MAX];
    switch(ndims) {
    case 1: 
      size[0] = dims[0];
      nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims, size);
      unsigned int centers[NRRD_DIM_MAX];
      centers[0] = nrrdCenterNode;
      nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);
      break;
      
    case 2: 
      {
	switch (sz_last_dim) {
	case 3: // Vector data
	case 6: // Tensor data
	  size[0] = sz_last_dim;
	  size[1] = dims[0];
	  size[2] = dims[1];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims+1, size);
	  break;
	  
	default: // treat the rest as Scalar data
	  size[0] = dims[0];
	  size[1] = dims[1];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims, size); 
	  break;
	};


	unsigned int centers[NRRD_DIM_MAX];
	centers[0] = nrrdCenterNode;
	centers[1] = nrrdCenterNode;
	nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);
      }
      break;
      
    case 3: 
      {
	switch (sz_last_dim) {
	case 3: // Vector data
	case 6: // Tensor data
	  size[0] = sz_last_dim;
	  size[1] = dims[0];
	  size[2] = dims[1];
	  size[3] = dims[2];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims+1, size);
	  break;
	  
	default: // treat the rest as Scalar data
	  size[0] = dims[0];
	  size[1] = dims[1];
	  size[2] = dims[2];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims, size);  
	  break;
	};

	unsigned int centers[NRRD_DIM_MAX];
	centers[0] = nrrdCenterNode;
	centers[1] = nrrdCenterNode;
	centers[2] = nrrdCenterNode;
	nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);
      }
      break;
      
    case 4: 
      {
	switch (sz_last_dim) {
	case 3: // Vector data
	case 6: // Tensor data
	  size[0] = sz_last_dim;
	  size[1] = dims[0];
	  size[2] = dims[1];
	  size[3] = dims[2];
	  size[4] = dims[3];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims+1, size);
	  break;
	  
	default: // treat the rest as Scalar data
	  size[0] = dims[0];
	  size[1] = dims[1];
	  size[2] = dims[2];
	  size[3] = dims[3];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims, size);  
	  break;
	};

	unsigned int centers[NRRD_DIM_MAX];
	centers[0] = nrrdCenterNode;
	centers[1] = nrrdCenterNode;
	centers[2] = nrrdCenterNode;
	centers[3] = nrrdCenterNode;
	
	nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);
      }
      break;
      
    case 5: 
      {
	switch (sz_last_dim) {
	case 3: // Vector data
	case 6: // Tensor data
	  size[0] = sz_last_dim;
	  size[1] = dims[0];
	  size[2] = dims[1];
	  size[3] = dims[2];
	  size[4] = dims[3];
	  size[5] = dims[4];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims+1, size);
	  break;
	  
	default: // treat the rest as Scalar data
	  size[0] = dims[0];
	  size[1] = dims[1];
	  size[2] = dims[2];
	  size[3] = dims[3];
	  size[4] = dims[4];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims, size);  
	  break;
	};

	unsigned int centers[NRRD_DIM_MAX];
	centers[0] = nrrdCenterNode;
	centers[1] = nrrdCenterNode;
	centers[2] = nrrdCenterNode;
	centers[3] = nrrdCenterNode;
	centers[4] = nrrdCenterNode;
	nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);
      }
      
      break;
      
    case 6: 
      {
	switch (sz_last_dim) {
	case 3: // Vector data
	case 6: // Tensor data
	  size[0] = sz_last_dim;
	  size[1] = dims[0];
	  size[2] = dims[1];
	  size[3] = dims[2];
	  size[4] = dims[3];
	  size[5] = dims[4];
	  size[6] = dims[5];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims+1, size);
	  break;
	  
	default: // treat the rest as Scalar data
	  size[0] = dims[0];
	  size[1] = dims[1];
	  size[2] = dims[2];
	  size[3] = dims[3];
	  size[4] = dims[4];
	  size[5] = dims[5];
	  nrrdWrap_nva(nout->nrrd_, data, nrrd_type, ndims, size);
	  break;
	};

	unsigned int centers[NRRD_DIM_MAX];
	centers[0] = nrrdCenterNode;
	centers[1] = nrrdCenterNode;
	centers[2] = nrrdCenterNode;
	centers[3] = nrrdCenterNode;
	centers[4] = nrrdCenterNode;
	centers[5] = nrrdCenterNode;
	nrrdAxisInfoSet_nva(nout->nrrd_, nrrdAxisInfoCenter, centers);
      }
      break;
    }
   
    string nrrdName;
    nrrdName.clear();

    // Remove the MDS characters that are illegal in Nrrds.
    const string nums("\"\\:.()");
	      
    for(string::size_type i = 0; i < signal.size(); i++) {
      bool in_set = false;
/*      for (unsigned int c = 0; c < nums.size(); c++) {
	if (signal[i] == nums[c]) {
	  in_set = true;
	  break;
	}
      }
*/
      
      if (in_set) { nrrdName.push_back('_' ); }
      else        { nrrdName.push_back(signal[i]); }
      
    }

    // Remove all of the tcl special characters.
    std::string::size_type pos;
    while( (pos = nrrdName.find("/")) != string::npos )
      nrrdName.replace( pos, 1, "-" );

    switch (sz_last_dim) {
    case 3: // Vector data
      nrrdName += ":Vector";
      nout->nrrd_->axis[0].kind = nrrdKind3Vector;
      break;
	  
    case 6: // Matrix data
      nrrdName += ":Matrix";
      nout->nrrd_->axis[0].kind = nrrdKind3DSymMatrix;
      break;
	  
    case 9: // Matrix data
      nrrdName += ":Matrix";
      nout->nrrd_->axis[0].kind = nrrdKind3DMatrix;
      break;
	  
    default: // treat the rest as Scalar data
      nrrdName += ":Scalar";
      nout->nrrd_->axis[0].kind = nrrdKindDomain;
      break;
    };

    for( int i=1; i<ndims; i++ )
      nout->nrrd_->axis[i].kind = nrrdKindDomain;


    nout->set_property( "Name", nrrdName, false );

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
  return NULL;

#endif
}

void MDSPlusDataReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("MDSPlusDataReader needs a minor command");
    return;
  }

  if (args[1] == "update_tree") {
#ifdef HAVE_MDSPLUS
    gui_loadServer_.reset();
    gui_loadTree_.reset();
    gui_loadShot_.reset();
    gui_loadSignal_.reset();

    // Get the load strings;
    string server( gui_loadServer_.get() );
    string tree  ( gui_loadTree_.get() );
    int    shot  ( atoi( gui_loadShot_.get().c_str() ) );
    string signal( gui_loadSignal_.get() );
    unsigned int depth = 1;

    if (args[2] == "root") {
      signal = string( gui_loadSignal_.get() );
    } else {
      signal = string( args[2] );
    }

    string dumpname;

    char* tmpdir = getenv( "SCIRUN_TMP_DIR" );
    char filename[64];

    sprintf( filename, "%s_%d.mds.dump", tree.c_str(), shot );

    if( tmpdir )
      dumpname = tmpdir + string( "/" );
    else
      dumpname = string( "/tmp/" );

    dumpname.append( filename );

    std::ofstream sPtr( dumpname.c_str() );

    if( !sPtr ) {
      error( string("Unable to open output file: ") + dumpname );
      get_gui()->execute( "reset_cursor" );
      return;
    }
  
    MDSPlusDump mdsdump( &sPtr );

    if( mdsdump.tree(server, tree, shot, signal, depth ) < 0 ) {
      error( mdsdump.error() );
      get_gui()->execute( "reset_cursor" );

      sPtr.flush();
      sPtr.close();

      return;
    }

    sPtr.flush();
    sPtr.close();

    // Update the treeview in the GUI.
    ostringstream str;
    str << get_id() << " build_tree " << dumpname << " " << args[3];
      
    get_gui()->execute(str.str().c_str());
#else

  error( "No MDS PLUS availible." );

#endif

  } else if (args[1] == "search") {
#ifdef HAVE_MDSPLUS
    gui_searchServer_.reset();
    gui_searchTree_.reset();
    gui_searchShot_.reset();
    gui_searchSignal_.reset();
    gui_searchPath_.reset();

    // Get the search strings;
    string server( gui_searchServer_.get() );
    string tree  ( gui_searchTree_.get() );
    int    shot  ( atoi( gui_searchShot_.get().c_str() ) );
    string signal( gui_searchSignal_.get() );
    int    path  ( gui_searchPath_.get() );

    MDSPlusReader mds;

    int trys = 0;
    int retVal = 0;

    while( trys < 10 && (retVal = mds.connect( server.c_str()) ) == -2 ) {
      remark( "Waiting for the connection to become free." );
      sleep( 1 );
      trys++;
    }

    /* Connect to MDSplus */
    if( retVal == -2 ) {
      error( "Connection to Mds Server " + server + " too busy ... giving up.");
      execute_error_ = true;
      return;
    }
    else if( retVal < 0 ) {
      error( "Connecting to Mds Server " + server );
      execute_error_ = true;
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
      execute_error_ = true;
      return;
    }
    if( retVal < 0 ) {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      error( str.str() );
      execute_error_ = true;
      return;
    }
    else {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      remark( str.str() );
    }

    vector< string > signals;

    {
      ostringstream str;
      str << "Searching " << signal << " for signals ... ";
      remark( str.str() );
    }

    mds.names( signal, signals, (bool) path, 1, 0 );

    mds.disconnect();

    {
      ostringstream str;
      str << signals.size() << " Signals loaded.. ";
      remark( str.str() );
    }


    for( unsigned int ic=0; ic<signals.size(); ic++ ) {

      // Update the list in the GUI.
      ostringstream str;
      str << get_id() << " setEntry {" << signals[ic] << "}";
      
      get_gui()->execute(str.str().c_str());
    }

#else

    error( "No MDS PLUS availible." );

#endif
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace DataIO

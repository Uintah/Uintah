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
 * C++ (CC) FILE : DicomToNrrd.cc
 *
 * DESCRIPTION   : This module allows an arbitrary number of DICOM series'
 *                 to be read in and converted to 4D nrrds for processing and
 *                 visualization.  The UI associated with this module provides
 *                 a mechanism for selecting DICOM series' and files within
 *                 those series'.
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *         
 * CREATED       : 9/19/2003
 * MODIFIED      : 10/3/2003
 * DOCUMENTATION :
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/

// SCIRun includes

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Teem/share/share.h>
#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>

#ifdef HAVE_INSIGHT
#include <Core/Algorithms/DataIO/DicomSeriesReader.h>
#include <Core/Algorithms/DataIO/DicomImage.h>
#endif

namespace SCITeem {

using namespace SCIRun;

// ****************************************************************************
// ***************************** Class: DicomToNrrd ***************************
// ****************************************************************************
 
struct series 
{
  string dir;
  string series_uid;
  vector<string> files;
};

class TeemSHARE DicomToNrrd : public Module 
{

public:

  // Constructor
  DicomToNrrd(GuiContext*);

  // Destructor
  virtual ~DicomToNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

#ifdef HAVE_INSIGHT

  int build_nrrds( vector<Nrrd*> & array );

  NrrdData * join_nrrds( vector<Nrrd*> arr );

#endif

private:

  //! GUI variables
  GuiString dir_;
  GuiString prefix_;
  GuiInt start_index_;
  GuiInt end_index_;
  GuiString series_uid_;
  GuiString series_files_;
  GuiString messages_;
  GuiString suid_sel_;  
  GuiString series_del_;  

  //! Ports
  NrrdOPort*      onrrd_;

  //! Set of Dicom series' 
  vector<struct series> all_series_;

};

DECLARE_MAKER(DicomToNrrd)

/*===========================================================================*/
// 
// DicomToNrrd
//
// Description : Constructor
//
// Arguments   :
//
// GuiContext* ctx - GUI context
//
DicomToNrrd::DicomToNrrd(GuiContext* ctx)
  : Module("DicomToNrrd", ctx, Source, "DataIO", "Teem"),
    dir_(ctx->subVar("dir")),
    prefix_(ctx->subVar("prefix")),
    start_index_(ctx->subVar("start-index")),
    end_index_(ctx->subVar("end-index")),
    series_uid_(ctx->subVar("series-uid")),    
    series_files_(ctx->subVar("series-files")),    
    messages_(ctx->subVar("messages")),    
    suid_sel_(ctx->subVar("suid-sel")),    
    series_del_(ctx->subVar("series-del"))    
{
}

/*===========================================================================*/
// 
// ~DicomToNrrd
//
// Description : Destructor
//
// Arguments   : none
//
DicomToNrrd::~DicomToNrrd(){
}


/*===========================================================================*/
// 
// execute 
//
// Description : The execute function for this module.  This is the control
//               center for the module.  This reads a series of DICOM files,
//               constructs a nrrd with the DICOM data, and sends the nrrd
//               downstream.
//
// Arguments   : none
//
void DicomToNrrd::execute(){

#ifdef HAVE_INSIGHT

  // If no DICOM series' were specified via the UI, print error and return
  if( all_series_.size() == 0 ) 
  {
    error("(DicomToNrrd::execute) No DICOM series' specified -- use 'UI' button to select series files.");
    return;
  }
  
  // Build a vector of nrrds from one or more DICOM series'
  vector<Nrrd*> arr;
  int ret;
  if( (ret = build_nrrds(arr)) == -1 )
  {
    return;
  }

  // Build single NrrdData object from all nrrds in vector
  NrrdData * sciNrrd = join_nrrds( arr );
  if( sciNrrd == 0 ) 
  {
    error("(DicomToNrrd::execute) Failed to join nrrds.");
    return;
  }

  // Create handle to data
  NrrdDataHandle sciNrrdHandle(sciNrrd);

  // Initialize output port
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");

  if( !onrrd_ ) {
    error("(DicomToNrrd::execute) Unable to initialize oport 'Nrrd'.");
    return;
  }

  // Send nrrd data downstream
  onrrd_->send(sciNrrdHandle);

  /*
  Since, in this case, nrrd didn't allocate the data, you might call
  "nrrdNix" to delete the nrrd struct, but not the data it points to
  (assuming someone else will deallocate it).  Or you can effectively
  hand the data off to nrrd and then delete it, along with the nrrd
  struct, with "nrrdNuke".
  */
  //nrrdNix(nrrd);

#else
  error("(DicomToNrrd::execute) Cannot read DICOM files.  Insight module needs to be included.");
  return;
#endif

}

/*===========================================================================*/
// 
// split_string
//
// Description :
//
// Arguments   :
//
// 
template < class T, class C >
void split_string(const T &src, C &container, typename T::value_type splitter)
{
  std::basic_istringstream<typename T::value_type> str_data(src);
  T line;
  while( std::getline(str_data, line, str_data.widen(splitter)) ) 
  {
    container.push_back(line);
  }
}

/*===========================================================================*/
// 
// tcl_command
//
// Description : The tcl_command function for this module.
//
// Arguments   :
//
// GuiArgs& args - GUI arguments
//
// void* userdata - ???
// 
void DicomToNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("DicomToNrrd needs a minor command");
    return;
  }

  if( args[1] == "get_series_uid" ) 
  {
#ifdef HAVE_INSIGHT

    DicomSeriesReader reader;
    dir_ = (ctx->subVar("dir"));
    string dir = dir_.get();

    reader.set_dir( dir );

    //cerr << "(DicomToNrrd::tcl_command) dir = " << dir << "\n";

    // Read all of the series uids from this directory, concatenate them to
    // form one string, and pass the string to the tcl side.
    std::vector<std::string> suids = reader.get_series_uids();
    int num_suids = suids.size();
    string all_suids = "";

    for( int i = 0; i < num_suids; i++ )
    {
      all_suids = string( all_suids + " " + suids[i] );  
    }

    if( num_suids == 0 )
    {
      string none_found = string( "No series' found in directory: " + dir );
      messages_.set( none_found );
    }

    //cerr << "(DicomToNrrd::tcl_command) all_suids = " << all_suids << "\n";

    series_uid_.set( all_suids );
#endif
  }
  else if( args[1] == "get_series_files" ) 
  {
#ifdef HAVE_INSIGHT
    DicomSeriesReader reader;
    dir_ = (ctx->subVar("dir"));
    string dir = dir_.get();
    reader.set_dir( dir );

    suid_sel_ = (ctx->subVar("suid-sel"));
    string suid = suid_sel_.get();

    //cerr << "(DicomToNrrd::tcl_command) suid = " << suid << "\n";

    std::vector<std::string> files = reader.get_file_names( suid );
    int num_files = files.size();
    string all_files = "";

    for( int i = 0; i < num_files; i++ )
    {
      all_files = string( all_files + " " + files[i] );  
    }

    //cerr << "(DicomToNrrd::tcl_command) all_files = " << all_files << "\n";
    series_files_.set( all_files );
#endif
  }
  else if( args[1] == "add_data" )
  {
#ifdef HAVE_INSIGHT
    dir_ = (ctx->subVar("dir"));
    string dir = dir_.get();

    series_files_ = (ctx->subVar("series-files"));
    string series_files = series_files_.get();

    suid_sel_ = (ctx->subVar("suid-sel"));
    string suid_sel = suid_sel_.get();

    // Create a new series
    struct series new_series;

    // Convert string of file names to vector of file names
    vector<string> files;
    split_string( series_files, files, ' ' );

    // First entry is always extra, chop it off

    new_series.dir = dir;
    new_series.series_uid = suid_sel;
    new_series.files = files;

    // Make sure this has the same number of files as all the other series'
    if( all_series_.size() > 0 ) 
    {
      if( new_series.files.size() != (all_series_[0]).files.size() )
      {
        warning( "(DicomToNrrd::tcl_command) Cannot load multiple series' with different numbers of files." );
      }
    }

    all_series_.insert( all_series_.end(), new_series );
#endif
  } 
  else if( args[1] == "delete_data" )
  {
#ifdef HAVE_INSIGHT
    // Get the selected series to be deleted
    series_del_ = (ctx->subVar("series-del"));
    string series_del = series_del_.get();

    // Split the series_del string by spaces
    vector<string> info;
    split_string( series_del, info, ' ' );

    //cerr << "info size = " << info.size() << "\n";

    if( info.size() < 17 ) {
      error("(DicomToNrrd::tcl_command) Delete series failed. Bad series info.");
      return;
    }

    // Get the directory 
    string dir = info[1];
    //cerr << "(DicomToNrrd::tcl_command) dir = " << dir << "\n";

    // Get the series uid
    string suid = info[6];
    //cerr << "(DicomToNrrd::tcl_command) suid = " << suid << "\n";

    // Get the start file
    string start_file = info[11];
    //cerr << "(DicomToNrrd::tcl_command) start_file = " << start_file << "\n";

    // Get the end file
    string end_file = info[16];
    //cerr << "(DicomToNrrd::tcl_command) end_file = " << end_file << "\n";

    // Find the matching entry in the all_series vector and remove it
    int num_series = all_series_.size();
    vector<struct series>::iterator iter = all_series_.begin();
     
    for( int i = 0; i < num_series; i++ )
    {
      struct series s = all_series_[i];
     
      if( (s.dir == dir) && (s.series_uid == suid) && 
          (s.files[0] == start_file) &&
          (s.files[s.files.size() - 1] == end_file) )
      {
        // Erase this element from the vector of series'
        all_series_.erase( iter );
      }  
      iter++;
    }
#endif
  }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}

#ifdef HAVE_INSIGHT

/*===========================================================================*/
// 
// build_nrrds 
//
// Description : Given a set of DICOM series', reads all series' in and builds
//               one nrrd object for each series.  Returns 0 on success, -1
//               on failure.
//
// Arguments   : 
//
// vector<Nrrd*> & array - Empty vector that will be initialized to contain
//                         one nrrd object per series.  
//
int DicomToNrrd::build_nrrds( vector<Nrrd*> & array )
{

  // Get the number of series' to build nrrds from 
  int num_series = all_series_.size();

  //cerr << "(DicomToNrrd::build_nrrds) num_series = " << num_series << "\n";
  //int num_series = 1;

  vector<Nrrd*> arr( num_series );
  DicomSeriesReader reader;

  // Read each DICOM series and build a nrrd object from it
  for( int i = 0; i < num_series; i++ )
  {
    // Get the struct for this series
    struct series ds = all_series_[i];

    DicomImage image;
    reader.set_dir( ds.dir );
    reader.set_files( ds.files );
    int ret;
    if( (ret = reader.read(image)) == -1 )
    {
      error("(DicomToNrrd::build_nrrds) DICOM read failed.");
      return -1;
    }

    image.print_image_info();
  
    // Construct a nrrd from the DICOM data
    char *err;
    Nrrd *nrrd = nrrdNew();

    // The arguments to nrrdWrap:
                                                                               
    // - "data" is a pointer to the raw, contiguous, pixel data.  The
    // parameter type is void*, but you'll be passing a float*, or a
    // double*, or something like that.                                        
                           
    // - I picked "nrrdTypeFloat" out of a hat; you should be choosing from the
    // nrrdType* enum values defined in nrrdEnums.h, to describe the
    // type that data points to.
                                                                               
    // - "3" is the dimension of field being wrapped in the example,
    // and it means that you should pass three numbers (sizeX, sizeY, sizeZ)
    // as the remaining arguments of nrrdWrap().  If this was 2D image data,
    // you pass only two arguments.

    // The nrrd struct does not try to remember "ownership" of the data you
    // give to it: nrrd->data will be set to whatever pointer you give me,
    // and I'll use it assuming that no one is going to free it on me.
    int dim = image.get_dimension();
    if( dim == 3 ) 
    {
      if( nrrdWrap(nrrd, image.get_pixel_buffer(), nrrdTypeUShort, 
               image.get_dimension(), image.get_size(0), 
               image.get_size(1), image.get_size(2)) ) 
      {
        error( "(DicomToNrrd::execute) Error creating nrrd." );
        err = biffGetDone(NRRD);
        // There was an error. "err" is a char* error message, pass it
        // to whatever kind of error handler you are using.  In case
        // you're picky about memory leaks, its up to you to:
        free(err);
      }
  
      nrrdAxisInfoSet( nrrd, nrrdAxisInfoCenter,
	    	       nrrdCenterNode, nrrdCenterNode, 
		       nrrdCenterNode, nrrdCenterNode );

      nrrd->axis[0].label = "Unknown:Scalar";
      nrrd->axis[1].label = strdup("x");
      nrrd->axis[2].label = strdup("y");
      nrrd->axis[3].label = strdup("z");
      nrrd->axis[1].spacing = image.get_spacing(0);
      nrrd->axis[2].spacing = image.get_spacing(1);
      nrrd->axis[3].spacing = image.get_spacing(2);
    }
    else if( dim == 2 ) 
    {
      if( nrrdWrap(nrrd, image.get_pixel_buffer(), nrrdTypeUShort, 
               image.get_dimension(), image.get_size(0), 
               image.get_size(1)) ) 
      {
        error( "(DicomToNrrd::execute) Error creating nrrd." );
        err = biffGetDone(NRRD);
        // There was an error. "err" is a char* error message, pass it
        // to whatever kind of error handler you are using.  In case
        // you're picky about memory leaks, its up to you to:
        free(err);
      }
  
      nrrdAxisInfoSet( nrrd, nrrdAxisInfoCenter,
	    	       nrrdCenterNode, nrrdCenterNode, 
		       nrrdCenterNode, nrrdCenterNode );

      nrrd->axis[0].label = "Unknown:Scalar";
      nrrd->axis[1].label = strdup("x");
      nrrd->axis[2].label = strdup("y");
      nrrd->axis[1].spacing = image.get_spacing(0);
      nrrd->axis[2].spacing = image.get_spacing(1);
    }
    else
    {
      error( "(DicomToNrrd::build_nrrds) Invalid image dimension." );
    }

    // Add this nrrd to the vector of nrrds
    arr[i] = nrrd;
  }

  array = arr;
  return 0;
}

/*===========================================================================*/
// 
// join_nrrds 
//
// Description : Given a vector of 3D nrrd objects, builds a single 4D nrrd
//               that is a combination of all the nrrds.  Uses the nrrdJoin
//               function to do this.  Returns the resulting NrrdData object
//               on success. Returns 0 on failure.
//
// Arguments   : 
//
// vector<Nrrd*> arr - Vector of pointers to nrrd objects. These nrrds should 
//                     all have the same dimensions. 
//
NrrdData * DicomToNrrd::join_nrrds( vector<Nrrd*> arr )
{
  int num_nrrds = arr.size();
  //cerr << "(DicomToNrrd::join_nrrds) num_nrrds = " << num_nrrds << "\n";

  if( num_nrrds == 0 )
  {
    error( "(DicomToNrrd::join_nrrds) No nrrds built." );
    return 0;
  }

  // Join all nrrds together into one 4D nrrd object
  NrrdData *sciNrrd = scinew NrrdData();
  sciNrrd->nrrd = nrrdNew();

  if( nrrdJoin(sciNrrd->nrrd, &arr[0], num_nrrds, 0, true) ) 
  {
    char *err = biffGetDone(NRRD);
    error( string("(DicomToNrrd::join_nrrds) Join Error: ") +  err );
    free(err);
    return 0;
  }

  nrrdAxisInfoSet(sciNrrd->nrrd, nrrdAxisInfoCenter,
			nrrdCenterNode, nrrdCenterNode, 
			nrrdCenterNode, nrrdCenterNode);

  string new_label("");
  for( int i = 0; i < num_nrrds; i++ )
  {
    if (i == 0) 
    {
      new_label += string((arr[i])->axis[0].label);
    } 
    else 
    {
      new_label += string(",") + string(arr[i]->axis[0].label);
    }
  }

  sciNrrd->nrrd->axis[0].label = strdup( new_label.c_str() );
  sciNrrd->nrrd->axis[1].label = strdup( "x" );
  sciNrrd->nrrd->axis[2].label = strdup( "y" );
  sciNrrd->nrrd->axis[3].label = strdup( "z" );
  sciNrrd->nrrd->axis[1].spacing = arr[0]->axis[1].spacing;
  sciNrrd->nrrd->axis[2].spacing = arr[0]->axis[2].spacing;
  sciNrrd->nrrd->axis[3].spacing = arr[0]->axis[3].spacing; 

  return sciNrrd;
}

#endif 

} // End namespace SCITeem

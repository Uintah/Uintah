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
 * C++ (CC) FILE : DicomNrrdReader.cc
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
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>

#include <sci_defs/insight_defs.h> 

#ifdef HAVE_INSIGHT
#include <Core/Algorithms/DataIO/DicomSeriesReader.h>
#include <Core/Algorithms/DataIO/DicomImage.h>
#endif

// Standard includes
#include <dirent.h>

namespace SCITeem { 

using namespace SCIRun;

// ****************************************************************************
// ***************************** Class: DicomNrrdReader ***************************
// ****************************************************************************
 
struct series 
{
  string dir;
  string series_uid;
  vector<string> files;
};

class TeemSHARE DicomNrrdReader : public Module 
{

public:

  //! Constructor
  DicomNrrdReader(GuiContext*);

  //! Destructor
  virtual ~DicomNrrdReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

#ifdef HAVE_INSIGHT

  int build_nrrds( vector<Nrrd*> & array );

  NrrdData * join_nrrds( vector<Nrrd*> arr );

#endif

private:

  void split_string( string src, vector<string> &container );
  void split_filenames( string src, vector<string> &container );

  //! GUI variables
  GuiInt have_insight_;
  GuiString dir_;
  GuiString series_uid_;
  GuiString series_files_;
  GuiString messages_;
  GuiString suid_sel_;  
  GuiString series_del_;
  GuiInt num_entries_;
  vector< GuiString* > entry_dir_;
  vector< GuiString* > entry_suid_;
  vector< GuiFilename* > entry_files_;


  //! Ports
  NrrdOPort*      onrrd_;

  //! Set of Dicom series' 
  vector<struct series> all_series_;

};

DECLARE_MAKER(DicomNrrdReader)

/*===========================================================================*/
// 
// DicomNrrdReader
//
// Description : Constructor
//
// Arguments   :
//
// GuiContext* ctx - GUI context
//
DicomNrrdReader::DicomNrrdReader(GuiContext* ctx)
  : Module("DicomNrrdReader", ctx, Source, "DataIO", "Teem"),
    have_insight_(ctx->subVar("have-insight", false)),
    dir_(ctx->subVar("dir")),
    series_uid_(ctx->subVar("series-uid")),    
    series_files_(ctx->subVar("series-files")),    
    messages_(ctx->subVar("messages")),    
    suid_sel_(ctx->subVar("suid-sel")),    
    series_del_(ctx->subVar("series-del")),
    num_entries_(ctx->subVar("num-entries"))
{
#ifdef HAVE_INSIGHT
  have_insight_.set(1);
#else
  have_insight_.set(0);
#endif
}

/*===========================================================================*/
// 
// ~DicomNrrdReader
//
// Description : Destructor
//
// Arguments   : none
//
DicomNrrdReader::~DicomNrrdReader(){
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
void DicomNrrdReader::execute(){

#ifdef HAVE_INSIGHT

  gui->execute(id + " sync_filenames");

  // If no DICOM series' were specified via the UI, print error and return
  if( all_series_.size() == 0 ) 
  {
    error("(DicomNrrdReader::execute) No DICOM series' specified -- use 'UI' button to select series files.");
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
    error("(DicomNrrdReader::execute) Failed to join nrrds.");
    return;
  }

  // Send nrrd data downstream.
  NrrdDataHandle sciNrrdHandle(sciNrrd);
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");
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
  error("(DicomNrrdReader::execute) Cannot read DICOM files.  Insight module needs to be included.");
  return;
#endif

}

/*===========================================================================*/
// 
// split_string
//
// Description : Splits a string into vector of strings where each string is 
//               space delimited in the original string.
//
// Arguments   :
//
// string src - String to be split.
// vector<string> &container - Vector of strings to contain result.
// 
void DicomNrrdReader::split_string( string src, vector<string> &container )
{
  // grab DIR string
  if (src.substr(0,5) != "DIR: ") {
    error("Error parsing entry for removal: no DIR tag");
    return;
  }
  container.push_back("DIR:");
  string temp = src.substr(5,src.length());

  // grab directory encapsulated in quotes
  int start = 0;
  int end = 1;
  start = temp.find("\"");
  end = temp.find("\"", end);
  container.push_back(temp.substr(start+1,end-1));

  if (end+4 <= (int)temp.length()) {
    temp = temp.substr(end+4, temp.length());
  }
  else {
    error("Directory string formated incorrectly.");
    return;
  }

  // grab SERIRES and UID
  if (temp.substr(0, 12) != "SERIES UID: ") {
    error("Error parsing entry for removal: no SERIES UID tag.");
    return;
  }

  container.push_back("SERIES");
  container.push_back("UID:");
  temp = temp.substr(12, temp.length());

  // grab series uid encapsulated in quotes
  start = 0;
  end = 1;
  start = temp.find("\"");
  end = temp.find("\"", end);
  container.push_back(temp.substr(start+1,end-1));

  if (end+4 <= (int)temp.length()) {
    temp = temp.substr(end+4, temp.length());
  }
  else {
    error("Series uid string formated incorrectly.");
    return;
  } 

  // grab START FILE:
  if (temp.substr(0, 12) != "START FILE: ") {
    error("Error parsing entry for removal: no START FILE tag");
    return;
  }
  container.push_back("START");
  container.push_back("FILE:");
  temp = temp.substr(12, temp.length());
  
  // grab start file encapsulated in quotes
  start = 0;
  end = 1;
  start = temp.find("\"");
  end = temp.find("\"", end);
  container.push_back(temp.substr(start+1,end-1));

  if (end+4 <= (int)temp.length()) {
    temp = temp.substr(end+4, temp.length());
  }
  else {
    error("Start file string formated incorrectly.");
    return;
  } 

  // grab END FILE:
  if (temp.substr(0, 10) != "END FILE: ") {
    error("Error parsing entry for removal: no END FILE tag");
    return;
  }
  container.push_back("END");
  container.push_back("FILE:");
  temp = temp.substr(10, temp.length());
  
  // grab end file encapsulated in quotes
  start = 0;
  end = 1;
  start = temp.find("\"");
  end = temp.find("\"", end);
  container.push_back(temp.substr(start+1,end-1));

  if (end+4 <= (int)temp.length()) {
    temp = temp.substr(end+4, temp.length());
  }
  else {
    error("Start file string formated incorrectly.");
    return;
  } 

  // grab NUMBER OF FILES:
  if (temp.substr(0,17) != "NUMBER OF FILES: ") {
    error("Error parsing entry for removal: no NUMBER OF FILEStag.");
    return;
  }
  container.push_back("NUMBER");
  container.push_back("OF");
  container.push_back("FILES:");
  temp = temp.substr(17, temp.length());

  // grab actual number
  container.push_back(temp.substr(0,temp.length()));
}

/*===========================================================================*/
// 
// split_filenames
//
// Description : Splits a string into vector of strings where each string is 
//               encapsulated withing curly braces in the original string.
//
// Arguments   :
//
// string src - String to be split.
// vector<string> &container - Vector of strings to contain result.
// 
void DicomNrrdReader::split_filenames( string src, vector<string> &container )
{
  int start = 0;
  int end = 0;
  string temp = src;
  while (temp.length() > 0) {
    start = src.find("{");
    end = src.find("}");
    container.push_back(temp.substr(start+1,end-1));
    if (end+2 <= (int)temp.length()) 
      temp = temp.substr(end+2, temp.length());
    else
      temp = "";
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
void DicomNrrdReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("DicomNrrdReader needs a minor command");
    return;
  }

  if( args[1] == "get_series_uid" ) 
  {
#ifdef HAVE_INSIGHT

    DicomSeriesReader reader;
    string dir = args[2];

    // Check to make sure the directory exists
    DIR *dirp;
    dirp = opendir( dir.c_str() );
    if (!dirp)
    {
      string no_dir = string( "No such directory: " + dir );
      messages_.set( no_dir );
      string all_suids = "";
      series_uid_.set( all_suids );
      return;
    }
    closedir(dirp);

    reader.set_dir( dir );

    //cerr << "(DicomNrrdReader::tcl_command) dir = " << dir << endl;

    // Read all of the series uids from this directory, concatenate them to
    // form one string, and pass the string to the tcl side.
    std::vector<std::string> suids = reader.get_series_uids();
    int num_suids = suids.size();
    string all_suids = "{} ";

    for( int i = 0; i < num_suids; i++ )
    {
      all_suids = string( all_suids + "{" + suids[i] + "}" + " ");  
    }

    if( num_suids == 0 )
    {
      string none_found = string( "No series' found in directory: " + dir );
      messages_.set( none_found );
    }

    //cerr << "(DicomNrrdReader::tcl_command) all_suids = " << all_suids << endl;

    series_uid_.set( all_suids );
#endif
  }
  else if( args[1] == "get_series_files" ) 
  {
#ifdef HAVE_INSIGHT
    DicomSeriesReader reader;
    string dir = args[2];
    reader.set_dir( dir );

    string suid = args[3];

    //cerr << "(DicomNrrdReader::tcl_command) suid = " << suid << endl;

    std::vector<std::string> files = reader.get_file_names( suid );
    int num_files = files.size();
    string all_files = "";

    // skip first entry because it is always empty
    for( int i = 0; i < num_files; i++ )
    {
      // re-ordered so that when selecting an entire series, 
      // it comes out the same as when using ImageJ
      //all_files = string( all_files + " " + files[i] );  
      all_files = string( "{" + files[i] +"}" + " " + all_files);  
    }
    //cerr << "(DicomNrrdReader::tcl_command) all_files = " << all_files << endl;
    series_files_.set( all_files );
#endif
  }
  else if( args[1] == "add_data" )
  {
#ifdef HAVE_INSIGHT

    string dir = args[2];

    string series_files = args[4];

    string suid_sel = args[3];

    // Create a new series
    struct series new_series;

    // Convert string of file names to vector of file names
    vector<string> files;
    split_filenames( series_files, files );

    // First entry is always extra, chop it off

    new_series.dir = dir;
    new_series.series_uid = suid_sel;
    new_series.files = files;

    // Make sure this has the same number of files as all the other series'
    if( all_series_.size() > 0 ) 
    {
      if( new_series.files.size() != (all_series_[0]).files.size() )
      {
        warning( "(DicomNrrdReader::tcl_command) Cannot load multiple series' with different numbers of files." );
      }
    }
    all_series_.insert( all_series_.end(), new_series );

    ostringstream str1;
    str1 << "entry-dir" << all_series_.size() - 1;
    entry_dir_.insert(entry_dir_.end(), new GuiString(ctx->subVar(str1.str())));

    ostringstream str2;
    str2 << "entry-suid" << all_series_.size() - 1;
    entry_suid_.insert(entry_suid_.end(), new GuiString(ctx->subVar(str2.str())));

    ostringstream str3;
    str3 << "entry-files" << all_series_.size() - 1;
    entry_files_.insert(entry_files_.end(), new GuiFilename(ctx->subVar(str3.str())));

#endif
  } 
  else if( args[1] == "delete_data" )
  {
#ifdef HAVE_INSIGHT
    // Get the selected series to be deleted
    string series_del = args[2];

    // Split the series_del string by spaces
    vector<string> info;
    split_string( series_del, info );

    //cerr << "info size = " << info.size() << endl;

    if( info.size() < 11 ) {
      error("(DicomNrrdReader::tcl_command) Delete series failed. Bad series info.");
      return;
    }

    // Get the directory 
    string dir = info[1];
    //cerr << "(DicomNrrdReader::tcl_command) dir = " << dir << endl;

    // Get the series uid
    string suid = info[4];
    //cerr << "(DicomNrrdReader::tcl_command) suid = " << suid << endl;

    // Get the start file
    string start_file = info[7];
    //cerr << "(DicomNrrdReader::tcl_command) start_file = " << start_file << endl;

    // Get the end file
    string end_file = info[10];
    //cerr << "(DicomNrrdReader::tcl_command) end_file = " << end_file << endl;

    // Find the matching entry in the all_series vector and remove it
    int num_series = all_series_.size();
    vector<struct series>::iterator iter = all_series_.begin();
    vector<GuiString*>::iterator iter2 = entry_dir_.begin();
    vector<GuiString*>::iterator iter3 = entry_suid_.begin();
    vector<GuiFilename*>::iterator iter4 = entry_files_.begin();
     
    for( int i = 0; i < num_series; i++ )
    {
      struct series s = all_series_[i];
     
      if( (s.dir == dir) && (s.series_uid == suid) && 
          (s.files[0] == start_file) &&
          (s.files[s.files.size() - 1] == end_file) )
      {
        // Erase this element from the vector of series'
        all_series_.erase( iter );

	// remove the guivar from the filenames
	entry_dir_.erase( iter2 );
	entry_suid_.erase( iter3 );
	entry_files_.erase( iter4 );
      }  
      iter++;
      iter2++;
      iter3++;
      iter4++;
    }
#endif
  }
  else if ( args[1] == "clear_data" )
    {
#ifdef HAVE_INSIGHT
      all_series_.clear();
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
int DicomNrrdReader::build_nrrds( vector<Nrrd*> & array )
{

  // Get the number of series' to build nrrds from 
  int num_series = all_series_.size();

  //cerr << "(DicomNrrdReader::build_nrrds) num_series = " << num_series << endl;
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
      error("(DicomNrrdReader::build_nrrds) DICOM read failed.  Check command line output for details.");
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
      if( nrrdWrap(nrrd, image.get_pixel_buffer(), nrrdTypeFloat, 
               3, image.get_size(0), 
               image.get_size(1), image.get_size(2)) ) 
      {
        error( "(DicomNrrdReader::execute) Error creating nrrd." );
        err = biffGetDone(NRRD);
        // There was an error. "err" is a char* error message, pass it
        // to whatever kind of error handler you are using.  In case
        // you're picky about memory leaks, its up to you to:
        free(err);
      }
  
      nrrdAxisInfoSet( nrrd, nrrdAxisInfoCenter,
	    	       nrrdCenterNode, nrrdCenterNode, 
		       nrrdCenterNode, nrrdCenterNode );

      // nrrd->axis[0].label = "Unknown:Scalar";
      nrrd->axis[0].label = strdup("x");
      nrrd->axis[1].label = strdup("y");
      nrrd->axis[2].label = strdup("z");
      nrrd->axis[0].spacing = image.get_spacing(0);
      nrrd->axis[1].spacing = image.get_spacing(1);
      nrrd->axis[2].spacing = image.get_spacing(2);
 
      nrrdAxisInfoMinMaxSet(nrrd, 0, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(nrrd, 1, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(nrrd, 2, nrrdCenterNode);
      //nrrdAxisInfoMinMaxSet(nrrd, 3, nrrdCenterNode);
    }
    else if( dim == 2 ) 
    {
      if( nrrdWrap(nrrd, image.get_pixel_buffer(), nrrdTypeFloat, 
               2, image.get_size(0), 
               image.get_size(1)) ) 
      {
        error( "(DicomNrrdReader::execute) Error creating nrrd." );
        err = biffGetDone(NRRD);
        // There was an error. "err" is a char* error message, pass it
        // to whatever kind of error handler you are using.  In case
        // you're picky about memory leaks, its up to you to:
        free(err);
      }
  
      nrrdAxisInfoSet( nrrd, nrrdAxisInfoCenter,
	    	       nrrdCenterNode, nrrdCenterNode, 
		       nrrdCenterNode, nrrdCenterNode );

      //nrrd->axis[0].label = "Unknown:Scalar";
      nrrd->axis[0].label = strdup("x");
      nrrd->axis[1].label = strdup("y");
      nrrd->axis[0].spacing = image.get_spacing(0);
      nrrd->axis[1].spacing = image.get_spacing(1);

      nrrdAxisInfoMinMaxSet(nrrd, 0, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(nrrd, 1, nrrdCenterNode);
      //nrrdAxisInfoMinMaxSet(nrrd, 2, nrrdCenterNode);
    }
    else
    {
      error( "(DicomNrrdReader::build_nrrds) Invalid image dimension." );
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
NrrdData * DicomNrrdReader::join_nrrds( vector<Nrrd*> arr )
{
  int num_nrrds = arr.size();
  //cerr << "(DicomNrrdReader::join_nrrds) num_nrrds = " << num_nrrds << endl;

  if( num_nrrds == 0 )
  {
    error( "(DicomNrrdReader::join_nrrds) No nrrds built." );
    return 0;
  }

  // Join all nrrds together into one 4D nrrd object
  NrrdData *sciNrrd = scinew NrrdData();
  sciNrrd->nrrd = nrrdNew();

  bool incr = true;
  if (num_nrrds == 1) { incr = false; }
  if( nrrdJoin(sciNrrd->nrrd, &arr[0], num_nrrds, 0, incr) ) 
  {
    char *err = biffGetDone(NRRD);
    error( string("(DicomNrrdReader::join_nrrds) Join Error: ") +  err );
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

  switch (sciNrrd->nrrd->dim) {
  case 4:
    if (incr) {
      sciNrrd->nrrd->axis[0].label = strdup( new_label.c_str() );
      sciNrrd->nrrd->axis[1].label = strdup( "x" );
      sciNrrd->nrrd->axis[2].label = strdup( "y" );
      sciNrrd->nrrd->axis[3].label = strdup( "z" );
      sciNrrd->nrrd->axis[1].spacing = arr[0]->axis[0].spacing;
      sciNrrd->nrrd->axis[2].spacing = arr[0]->axis[1].spacing;
      sciNrrd->nrrd->axis[3].spacing = arr[0]->axis[2].spacing; 
      
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 0, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 1, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 2, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 3, nrrdCenterNode);
    } else {
      return 0;
    }
    
    break;
  case 3:
    if (incr) {
      sciNrrd->nrrd->axis[0].label = strdup( new_label.c_str() );
      sciNrrd->nrrd->axis[1].label = strdup( "x" );
      sciNrrd->nrrd->axis[2].label = strdup( "y" );
      sciNrrd->nrrd->axis[1].spacing = arr[0]->axis[0].spacing;
      sciNrrd->nrrd->axis[2].spacing = arr[0]->axis[1].spacing;
    } else {
      sciNrrd->nrrd->axis[0].label = strdup( "x" );
      sciNrrd->nrrd->axis[1].label = strdup( "y" );
      sciNrrd->nrrd->axis[2].label = strdup( "z" );
      sciNrrd->nrrd->axis[0].spacing = arr[0]->axis[0].spacing; 
      sciNrrd->nrrd->axis[1].spacing = arr[0]->axis[1].spacing;
      sciNrrd->nrrd->axis[2].spacing = arr[0]->axis[2].spacing;
    }
    
    nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 0, nrrdCenterNode);
    nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 1, nrrdCenterNode);
    nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 2, nrrdCenterNode);
    break;
  case 2:
    if (incr) {
      return 0;
    } else {
      sciNrrd->nrrd->axis[0].label = strdup( "x" );
      sciNrrd->nrrd->axis[1].label = strdup( "y" );
      sciNrrd->nrrd->axis[0].spacing = arr[0]->axis[0].spacing; 
      sciNrrd->nrrd->axis[1].spacing = arr[0]->axis[1].spacing;

      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 0, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd, 1, nrrdCenterNode);
    }
    break;
  default:
    return 0;
  }
  return sciNrrd;
}

#endif 

} // End namespace SCITeem

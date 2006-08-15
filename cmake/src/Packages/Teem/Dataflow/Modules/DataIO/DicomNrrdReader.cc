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
 *                 12/29/2005 Darby Van Uitert
 * DOCUMENTATION :
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/

// SCIRun includes
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>

#include <sci_defs/teem_defs.h> 

#ifdef HAVE_GDCM
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmSerieHelper.h"
#endif

// Standard includes
#ifdef _WIN32
#include <Core/OS/Dir.h>
#else
#include <dirent.h>
#endif
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

class DicomNrrdReader : public Module 
{

public:

  //! Constructor
  DicomNrrdReader(GuiContext*);

  //! Destructor
  virtual ~DicomNrrdReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  Nrrd* load_single_file(const std::string &file);
  int build_nrrds( vector<Nrrd*> & array );

  NrrdData * join_nrrds( vector<Nrrd*> arr );

private:

  void split_string( string src, vector<string> &container );
  void split_filenames( string src, vector<string> &container );

  //! GUI variables
  GuiInt have_gdcm_;
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
  double z_spacing_;


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
    have_gdcm_(get_ctx()->subVar("have-gdcm", false)),
    dir_(get_ctx()->subVar("dir")),
    series_uid_(get_ctx()->subVar("series-uid"), ""),
    series_files_(get_ctx()->subVar("series-files"), ""),
    messages_(get_ctx()->subVar("messages"), ""),
    suid_sel_(get_ctx()->subVar("suid-sel"), ""),
    series_del_(get_ctx()->subVar("series-del"), ""),
    num_entries_(get_ctx()->subVar("num-entries"), 0),
    z_spacing_(0)
{
#ifdef HAVE_GDCM
  have_gdcm_.set(1);
#else
  have_gdcm_.set(0);
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
void DicomNrrdReader::execute()
{
#ifdef HAVE_GDCM

  get_gui()->execute(get_id() + " sync_filenames");
  
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
  onrrd_->send_and_dereference(sciNrrdHandle);

  /*
  Since, in this case, nrrd didn't allocate the data, you might call
  "nrrdNix" to delete the nrrd struct, but not the data it points to
  (assuming someone else will deallocate it).  Or you can effectively
  hand the data off to nrrd and then delete it, along with the nrrd
  struct, with "nrrdNuke".
  */
  //nrrdNix(nrrd);

#else
  error("(DicomNrrdReader::execute) Cannot read DICOM files.  SCIRun must be built with either the GDCM library or the Insight Package.");
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
void
DicomNrrdReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("DicomNrrdReader needs a minor command");
    return;
  }

  if( args[1] == "get_series_uid" ) 
  {

#ifdef HAVE_GDCM

    string dir = args[2];

#if ((GDCM_MAJOR_VERSION == 1) && (GDCM_MINOR_VERSION > 2)) || (GDCM_MAJOR_VERSION > 1)
    gdcm::SerieHelper *reader = gdcm::SerieHelper::New();
    reader->SetLoadMode(gdcm::LD_ALL);

    // Check to make sure the directory exists
    DIR *dirp;
    dirp = opendir( dir.c_str() );
    if (!dirp)
    {
      string no_dir = string( "No such directory: " + dir );
      messages_.set( no_dir );
      string all_suids = "";
      series_uid_.set( all_suids );
      reader->Delete();
      return;
    }
    closedir(dirp);

    reader->SetDirectory(dir, false);

    // Read all of the series uids from this directory, concatenate them to
    // form one string, and pass the string to the tcl side.
    // //std::vector<std::string> suids = reader.get_series_uids();
    // //int num_suids = suids.size();

#else
    gdcm::SerieHelper *reader = new gdcm::SerieHelper();
    reader->SetUseSeriesDetails( true );

    // Check to make sure the directory exists
    DIR *dirp;
    dirp = opendir( dir.c_str() );
    if (!dirp)
    {
      string no_dir = string( "No such directory: " + dir );
      messages_.set( no_dir );
      string all_suids = "";
      series_uid_.set( all_suids );
      delete reader;
      return;
    }
    closedir(dirp);

    reader->SetDirectory(dir, false);

    // Read all of the series uids from this directory, concatenate them to
    // form one string, and pass the string to the tcl side.
    // //std::vector<std::string> suids = reader.get_series_uids();
    // //int num_suids = suids.size();
#endif


    string all_suids = "{} ";
    std::vector<string> suids;

#if ((GDCM_MAJOR_VERSION == 1) && (GDCM_MINOR_VERSION > 2)) || (GDCM_MAJOR_VERSION > 1) 
    gdcm::FileList *l = reader->GetFirstSingleSerieUIDFileSet();
    
    while (l) 
    {
      string uid = reader->GetCurrentSerieUIDFileSetUID();
      char *temp = new char[uid.size()];
      memcpy(temp,uid.c_str(), uid.size());
      suids.push_back(temp);
      l = reader->GetNextSingleSerieUIDFileSet();
      delete [] temp;
    }
#else
    gdcm::GdcmFileList *l = reader->GetFirstCoherentFileList();
    
    while (l) 
    {
      gdcm::File *file = (*l)[0];
      string uid = reader->CreateUniqueSeriesIdentifier( file ).c_str();
      char *temp = new char[uid.size()];
      memcpy(temp,uid.c_str(), uid.size());
      suids.push_back(temp);
      l = reader->GetNextCoherentFileList();
      delete [] temp;
    }
#endif

    for(int i=0; i<(int)suids.size(); i++) {     
      all_suids = string( all_suids + "{" + suids[i] + "}" + " ");
    }

    series_uid_.set( all_suids );

    delete reader;
#endif
  }
  else if( args[1] == "get_series_files" ) 
  {
#ifdef HAVE_GDCM

    string dir = args[2];
    string suid = args[3];
    std::vector<std::string> files;


#if ((GDCM_MAJOR_VERSION == 1) && (GDCM_MINOR_VERSION > 2)) || (GDCM_MAJOR_VERSION > 1) 
    gdcm::SerieHelper *reader = gdcm::SerieHelper::New();
    reader->SetLoadMode(gdcm::LD_ALL);
    reader->SetDirectory(dir, false);

    gdcm::FileList *l = reader->GetFirstSingleSerieUIDFileSet();
    while (l) 
    {
      string cur_suid = reader->GetCurrentSerieUIDFileSetUID();
      char *temp = new char[cur_suid.size()];
      memcpy(temp, cur_suid.c_str(), cur_suid.size());
      if (suid.compare(temp) == 0) {
	for (std::vector<gdcm::File* >::iterator it =  l->begin();
	     it != l->end(); ++it) {
	  files.push_back((*it)->GetFileName());
	} 
	break;
      }
      delete [] temp;
      l = reader->GetNextSingleSerieUIDFileSet();
    }

#else
    gdcm::SerieHelper *reader = new gdcm::SerieHelper();
    reader->SetUseSeriesDetails( true );
    reader->SetDirectory(dir, false);

    gdcm::GdcmFileList *l = reader->GetFirstCoherentFileList();
    while (l) 
    {
      gdcm::File *file = (*l)[0];
      string cur_suid = reader->CreateUniqueSeriesIdentifier( file ).c_str();
      char *temp = new char[cur_suid.size()];
      memcpy(temp, cur_suid.c_str(), cur_suid.size());
      if (suid.compare(temp) == 0) {
	for (std::vector<gdcm::File* >::iterator it =  l->begin();
	     it != l->end(); ++it) {
	  files.push_back((*it)->GetFileName());
	} 
	break;
      }
      delete [] temp;
      l = reader->GetNextCoherentFileList();
    }
#endif


    string all_files = "";
    
    for(int i=0; i<(int)files.size();i ++) {
      all_files = string( "{" + files[i] + "}" + " " + all_files);
    }

    series_files_.set( all_files );

    delete reader;
#endif
  }
  else if( args[1] == "add_data" )
  {
#ifdef HAVE_GDCM
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
    entry_dir_.insert(entry_dir_.end(), new GuiString(get_ctx()->subVar(str1.str())));

    ostringstream str2;
    str2 << "entry-suid" << all_series_.size() - 1;
    entry_suid_.insert(entry_suid_.end(), new GuiString(get_ctx()->subVar(str2.str())));

    ostringstream str3;
    str3 << "entry-files" << all_series_.size() - 1;
    entry_files_.insert(entry_files_.end(), new GuiFilename(get_ctx()->subVar(str3.str())));

#endif
  } 
  else if( args[1] == "delete_data" )
  {
    // Get the selected series to be deleted
    string series_del = args[2];

    // Split the series_del string by spaces
    vector<string> info;
    split_string( series_del, info );

    if( info.size() < 11 ) {
      error("(DicomNrrdReader::tcl_command) Delete series failed. Bad series info.");
      return;
    }

    // Get the directory 
    string dir = info[1];

    // Get the series uid
    string suid = info[4];

    // Get the start file
    string start_file = info[7];

    // Get the end file
    string end_file = info[10];

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
  }
  else if ( args[1] == "clear_data" )
    {
      all_series_.clear();
    }
  else 
  {
    Module::tcl_command(args, userdata);
  }
}


/*===========================================================================*/
// 
// load_single_file
//
// Description : Given a filename as a string, load it using GDCM and store
//               it in a nrrd.
//
// Arguments   : 
//
// const string & file - string containing filename to load in
//
Nrrd* DicomNrrdReader::load_single_file( const string &file )
{
#ifdef HAVE_GDCM

  string pixelType;
  size_t nrrd_size = 0;
  unsigned char *source = 0;

#if ((GDCM_MAJOR_VERSION == 1) && (GDCM_MINOR_VERSION > 2)) || (GDCM_MAJOR_VERSION > 1) 
  gdcm::File *f = gdcm::File::New();
  f->SetLoadMode(gdcm::LD_ALL);
  f->SetFileName( file );

  bool result = f->Load();
  if ( !result ) {
    error("Cannot process file.");
    f->Delete();
    return 0;
  }

  gdcm::FileHelper *fh = gdcm::FileHelper::New(f);
  if (!fh) {
    error("Failure to create GDCM FileHelper.");
    f->Delete();
    return 0;
  }

  // grab the sizes, spacings and origin for each dimensions
  int nX, nY, nZ;
  double sX, sY, sZ;
  double oX, oY, oZ;
  nX = f->GetXSize();
  nY = f->GetYSize();
  nZ = f->GetZSize();

  sX = f->GetXSpacing();
  sY = f->GetYSpacing();
  sZ = f->GetZSpacing();

  oX = f->GetXOrigin();
  oY = f->GetYOrigin();
  oZ = f->GetZOrigin();
  
  // grab the pixel type
  pixelType = f->GetPixelType();

  // get the pointer to the chunk of data that GDCM allocated
  source = (unsigned char*)fh->GetImageData();

  // Since GDCM owns the source data and will delete it when it wants
  // to, copy the data into a nrrd so nrrd can manage the data.
  Nrrd *nrrd = nrrdNew();

  int num_comp = f->GetNumberOfScalarComponents();

  unsigned int pType = nrrdTypeUnknown;
  unsigned int pType_size = 0;
  if (pixelType == "8U") {
    // unsigned char
    pType = nrrdTypeUChar;
    pType_size = sizeof(unsigned char);
  } else if (pixelType == "8S") {
    // char
    pType = nrrdTypeChar;
    pType_size = sizeof(char);
  } else if (pixelType == "16U") {
    // unsigned short
    pType = nrrdTypeUShort;
    pType_size = sizeof(unsigned short);
  } else if (pixelType == "16S") {
    // short
    pType = nrrdTypeShort;
    pType_size = sizeof(short);
  } else if (pixelType == "32U") {
    // unsigned int
    pType = nrrdTypeUInt;
    pType_size = sizeof(unsigned int);
  } else if (pixelType == "32S") {
    // int
    pType = nrrdTypeInt;
    pType_size = sizeof(int);
  } else if (pixelType == "FD") {
    // 64 bit double
    pType = nrrdTypeDouble;
    pType_size = sizeof(double);
  } else {
    error("Unrecognized pixel type\n");
    fh->Delete();
    f->Delete();
    return 0;
  }

  // nrrdAlloc and calculate nrrd_size
  // determine if it is scalar or vector data first
  if (num_comp == 1) {
    // determine if it is 2D, or 3D
    if (nZ > 1) {
      // 3D
      size_t size[NRRD_DIM_MAX];
      size[0] = nX; size[1] = nY; size[2] = nZ;
      nrrdAlloc_nva(nrrd, pType, 3, size);
      unsigned int centers[NRRD_DIM_MAX];
      centers[0] = nrrdCenterNode; centers[1] = nrrdCenterNode;
      centers[2] = nrrdCenterNode;
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKindDomain;
      nrrd->axis[1].kind = nrrdKindDomain;
      nrrd->axis[2].kind = nrrdKindDomain;
    } else {
      // 2D
      size_t size[NRRD_DIM_MAX];
      size[0] = nX; size[1] = nY;
      nrrdAlloc_nva(nrrd, pType, 2, size);
      unsigned int centers[NRRD_DIM_MAX];
      centers[0] = nrrdCenterNode; centers[1] = nrrdCenterNode;
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKindDomain;
      nrrd->axis[1].kind = nrrdKindDomain;
    }
  } else {
     // determine if it is 2D, or 3D vector data
    if (nZ > 1) {
      // 4D
      size_t size[NRRD_DIM_MAX];
      size[0] = num_comp; size[1] = nX; 
      size[2] = nY; size[3] = nZ;
      nrrdAlloc_nva(nrrd, pType, 4, size);
      unsigned int centers[4] = {nrrdCenterUnknown, nrrdCenterNode,
				 nrrdCenterNode, nrrdCenterNode};
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKind3Vector;
      nrrd->axis[1].kind = nrrdKindDomain;
      nrrd->axis[2].kind = nrrdKindDomain;
      nrrd->axis[3].kind = nrrdKindDomain;
    } else {
      // 3D
      size_t size[NRRD_DIM_MAX];
      size[0] = num_comp; size[1] = nX; size[2] = nY;
      nrrdAlloc_nva(nrrd, pType, 3, size);
      unsigned int centers[3] = {nrrdCenterUnknown, nrrdCenterNode, 
				 nrrdCenterNode};
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKind3Vector;
      nrrd->axis[1].kind = nrrdKindDomain;
      nrrd->axis[2].kind = nrrdKindDomain;
    }
  }

  nrrd_size = num_comp * nX * nY * nZ * pType_size;

  // only do the memcpy if the sizes of the buffers
  // are the same
  if (nrrd_size != fh->GetImageDataSize()) {
    error("Size of data allocated by NRRD does not equal the size allocated by GDCM");
    fh->Delete();
    f->Delete();
    return 0;
  }
  
  memcpy(nrrd->data, source, fh->GetImageDataSize());

  // FIX ME set NRRD properties (especially orientation stuff)
  // spacing, origin
  
  // FIX ME min, max, & spacing or spaceDirection vectors
  nrrd->axis[0].spacing = sX;
  nrrd->axis[1].spacing = sY;
  z_spacing_ = sZ;

  if (nrrd->dim > 2)
    nrrd->axis[2].spacing = sZ;

  nrrd->axis[0].label = airStrdup("x");
  nrrd->axis[1].label = airStrdup("y");
  if (nrrd->dim > 2)
    nrrd->axis[2].label = airStrdup("z");

  nrrd->axis[0].min = oX;
  nrrd->axis[0].max = oX + ((nrrd->axis[0].size - 1) * sX);
  nrrd->axis[1].min = oY;
  nrrd->axis[1].max = oY + ((nrrd->axis[1].size - 1) * sY);
  if (nrrd->dim > 2) {
    nrrd->axis[2].min = oZ;
    nrrd->axis[2].max = oZ + ((nrrd->axis[2].size - 1) * sZ);
  }

  // delete GDCM objects
  fh->Delete();
  f->Delete();

  return nrrd;
#else
  gdcm::File *f = new gdcm::File();
  f->SetFileName( file );

  bool result = f->Load();
  if ( !result ) {
    error("Cannot process file.");
    delete f;
    return 0;
  }

  gdcm::FileHelper *fh = new gdcm::FileHelper(f);
  if (!fh) {
    error("Failure to create GDCM FileHelper.");
    delete f;
    return 0;
  }

  // grab the sizes, spacings and origin for each dimensions
  int nX, nY, nZ;
  double sX, sY, sZ;
  double oX, oY, oZ;
  nX = f->GetXSize();
  nY = f->GetYSize();
  nZ = f->GetZSize();

  sX = f->GetXSpacing();
  sY = f->GetYSpacing();
  sZ = f->GetZSpacing();

  oX = f->GetXOrigin();
  oY = f->GetYOrigin();
  oZ = f->GetZOrigin();
  
  // grab the pixel type
  pixelType = f->GetPixelType();

  // get the pointer to the chunk of data that GDCM allocated
  source = (unsigned char*)fh->GetImageData();

  // Since GDCM owns the source data and will delete it when it wants
  // to, copy the data into a nrrd so nrrd can manage the data.
  Nrrd *nrrd = nrrdNew();

  int num_comp = f->GetNumberOfScalarComponents();

  unsigned int pType = nrrdTypeUnknown;
  unsigned int pType_size = 0;
  if (pixelType == "8U") {
    // unsigned char
    pType = nrrdTypeUChar;
    pType_size = sizeof(unsigned char);
  } else if (pixelType == "8S") {
    // char
    pType = nrrdTypeChar;
    pType_size = sizeof(char);
  } else if (pixelType == "16U") {
    // unsigned short
    pType = nrrdTypeUShort;
    pType_size = sizeof(unsigned short);
  } else if (pixelType == "16S") {
    // short
    pType = nrrdTypeShort;
    pType_size = sizeof(short);
  } else if (pixelType == "32U") {
    // unsigned int
    pType = nrrdTypeUInt;
    pType_size = sizeof(unsigned int);
  } else if (pixelType == "32S") {
    // int
    pType = nrrdTypeInt;
    pType_size = sizeof(int);
  } else if (pixelType == "FD") {
    // 64 bit double
    pType = nrrdTypeDouble;
    pType_size = sizeof(double);
  } else {
    error("Unrecognized pixel type\n");
    delete fh;
    delete f;
    return 0;
  }

  // nrrdAlloc and calculate nrrd_size
  // determine if it is scalar or vector data first
  if (num_comp == 1) {
    // determine if it is 2D, or 3D
    if (nZ > 1) {
      // 3D
      size_t size[NRRD_DIM_MAX];
      size[0] = nX; size[1] = nY; size[2] = nZ;
      nrrdAlloc_nva(nrrd, pType, 3, size);
      unsigned int centers[NRRD_DIM_MAX];
      centers[0] = nrrdCenterNode; centers[1] = nrrdCenterNode;
      centers[2] = nrrdCenterNode;
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKindDomain;
      nrrd->axis[1].kind = nrrdKindDomain;
      nrrd->axis[2].kind = nrrdKindDomain;
    } else {
      // 2D
      size_t size[NRRD_DIM_MAX];
      size[0] = nX; size[1] = nY;
      nrrdAlloc_nva(nrrd, pType, 2, size);
      unsigned int centers[NRRD_DIM_MAX];
      centers[0] = nrrdCenterNode; centers[1] = nrrdCenterNode;
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKindDomain;
      nrrd->axis[1].kind = nrrdKindDomain;
    }
  } else {
     // determine if it is 2D, or 3D vector data
    if (nZ > 1) {
      // 4D
      size_t size[NRRD_DIM_MAX];
      size[0] = num_comp; size[1] = nX;
      size[2] = nY; size[3] = nZ;
      nrrdAlloc_nva(nrrd, pType, 4, size);
      unsigned int centers[NRRD_DIM_MAX];
      centers[0] = nrrdCenterUnknown; centers[1] =  nrrdCenterNode;
      centers[2] = nrrdCenterNode; centers[3] = nrrdCenterNode;
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKind3Vector;
      nrrd->axis[1].kind = nrrdKindDomain;
      nrrd->axis[2].kind = nrrdKindDomain;
      nrrd->axis[3].kind = nrrdKindDomain;
    } else {
      // 3D
      size_t size[NRRD_DIM_MAX];
      size[0] = num_comp; size[1] = nX;
      size[2] = nY;
      nrrdAlloc_nva(nrrd, pType, 3, size);
      unsigned int centers[NRRD_DIM_MAX];
      centers[0] = nrrdCenterUnknown; centers[1] = nrrdCenterNode;
      centers[2] = nrrdCenterNode;
      nrrdAxisInfoSet_nva(nrrd, nrrdAxisInfoCenter, centers);
      nrrd->axis[0].kind = nrrdKind3Vector;
      nrrd->axis[1].kind = nrrdKindDomain;
      nrrd->axis[2].kind = nrrdKindDomain;
    }
  }

  nrrd_size = num_comp * nX * nY * nZ * pType_size;

  // only do the memcpy if the sizes of the buffers
  // are the same
  if (nrrd_size != fh->GetImageDataSize()) {
    error("Size of data allocated by NRRD does not equal the size allocated by GDCM");
    delete fh;
    delete f;
    return 0;
  }
  
  memcpy(nrrd->data, source, fh->GetImageDataSize());

  // FIX ME set NRRD properties (especially orientation stuff)
  // spacing, origin
  
  // FIX ME min, max, & spacing or spaceDirection vectors
  nrrd->axis[0].spacing = sX;
  nrrd->axis[1].spacing = sY;
  z_spacing_ = sZ;

  if (nrrd->dim > 2)
    nrrd->axis[2].spacing = sZ;

  nrrd->axis[0].label = airStrdup("x");
  nrrd->axis[1].label = airStrdup("y");
  if (nrrd->dim > 2)
    nrrd->axis[2].label = airStrdup("z");

  nrrd->axis[0].min = oX;
  nrrd->axis[0].max = oX + ((nrrd->axis[0].size - 1) * sX);
  nrrd->axis[1].min = oY;
  nrrd->axis[1].max = oY + ((nrrd->axis[1].size - 1) * sY);
  if (nrrd->dim > 2) {
    nrrd->axis[2].min = oZ;
    nrrd->axis[2].max = oZ + ((nrrd->axis[2].size - 1) * sZ);
  }

  // delete GDCM objects
  delete fh;
  delete f;

  return nrrd;
#endif


#else
  return 0;
#endif
}

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

  vector<Nrrd*> arr( num_series );

#ifdef HAVE_GDCM
  

  // Read each DICOM series and build a nrrd object from it
  for( int i = 0; i < num_series; i++ )
  {
    // Get the struct for this series
    struct series ds = all_series_[i];

    // For each file, read it in, create a nrrd, and stick that nrrd in 
    // the array. GDCM controls the pointer to the data and destroying 
    // it so allocate our own memory for the nrrd.
    vector<Nrrd *> slices(ds.files.size());

    for(int j=0; j<(int)ds.files.size(); j++) {
      Nrrd *slice = load_single_file( string(ds.dir + "/" + ds.files[j]) );
      if ( slice !=0 )
	slices[j] = slice;
      else
	error("NRRD from loading single file empty.");
    }

    if (slices.size() == 1) {
      arr[i] = slices[0];
    } else {
      // FIX ME: join all the slices and stick in arr[i]
      Nrrd *nrrd = nrrdNew();
      if( nrrdJoin(nrrd, &slices[0], slices.size(), slices[0]->dim, true) ) 
	{
	  char *err = biffGetDone(NRRD);
	error( string("(DicomNrrdReader::join_nrrds) Join Error: ") +  err );
	free(err);
	return 0;
	}
      
      arr[i] = nrrd;
    }
  }

#endif
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

  if( num_nrrds == 0 )
  {
    error( "(DicomNrrdReader::join_nrrds) No nrrds built." );
    return 0;
  }

  // Join all nrrds together into one 4D nrrd object
  NrrdData *sciNrrd = 0;

  bool incr = true;
  if (num_nrrds == 1) { incr = false; }

  int position = 0;  
  if (arr[0]->dim == 2) {
    position = 2;
  }

  if (arr.size() > 1) {
    sciNrrd = scinew NrrdData();
    sciNrrd->nrrd_ = nrrdNew();

    if( nrrdJoin(sciNrrd->nrrd_, &arr[0], num_nrrds, position, incr) ) 
      {
	char *err = biffGetDone(NRRD);
	error( string("(DicomNrrdReader::join_nrrds) Join Error: ") +  err );
	free(err);
	return 0;
      }
  } else {
    sciNrrd = scinew NrrdData(arr[0]);
  }

  unsigned int *centers = (unsigned int *)malloc(sizeof(unsigned int) * 
						 sciNrrd->nrrd_->dim);
  for(unsigned int i=0; i<sciNrrd->nrrd_->dim; i++)
    centers[i] = nrrdCenterNode;
  nrrdAxisInfoSet_nva(sciNrrd->nrrd_, nrrdAxisInfoCenter, &centers);
		      
  sciNrrd->nrrd_->axis[2].spacing = z_spacing_;


  if (arr[0]->dim == 2)
    incr = false;

  switch (sciNrrd->nrrd_->dim) {
  case 4:
    if (incr) {
      if (sciNrrd->nrrd_->axis[0].size > 4)
	sciNrrd->nrrd_->axis[0].label = airStrdup( "TensorData" );
      else
	sciNrrd->nrrd_->axis[0].label = airStrdup( "VectorData" );
      sciNrrd->nrrd_->axis[1].label = airStrdup( "x" );
      sciNrrd->nrrd_->axis[2].label = airStrdup( "y" );
      sciNrrd->nrrd_->axis[3].label = airStrdup( "z" );
      sciNrrd->nrrd_->axis[1].spacing = arr[0]->axis[0].spacing;
      sciNrrd->nrrd_->axis[2].spacing = arr[0]->axis[1].spacing;
      //sciNrrd->nrrd_->axis[3].spacing = arr[0]->axis[2].spacing; 
      sciNrrd->nrrd_->axis[3].spacing = z_spacing_; 
      
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 0, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 1, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 2, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 3, nrrdCenterNode);
    } else {
      return 0;
    }
    
    break;
  case 3:
    if (incr) {
      if (sciNrrd->nrrd_->axis[0].size > 4)
	sciNrrd->nrrd_->axis[0].label = airStrdup( "TensorData" );
      else
	sciNrrd->nrrd_->axis[0].label = airStrdup( "VectorData" );
      sciNrrd->nrrd_->axis[1].label = airStrdup( "x" );
      sciNrrd->nrrd_->axis[2].label = airStrdup( "y" );
      sciNrrd->nrrd_->axis[1].spacing = arr[0]->axis[0].spacing;
      sciNrrd->nrrd_->axis[2].spacing = arr[0]->axis[1].spacing;
    } else {
      sciNrrd->nrrd_->axis[0].label = airStrdup( "x" );
      sciNrrd->nrrd_->axis[1].label = airStrdup( "y" );
      sciNrrd->nrrd_->axis[2].label = airStrdup( "z" );
      sciNrrd->nrrd_->axis[0].spacing = arr[0]->axis[0].spacing; 
      sciNrrd->nrrd_->axis[1].spacing = arr[0]->axis[1].spacing;
      //sciNrrd->nrrd_->axis[2].spacing = arr[0]->axis[2].spacing;
      sciNrrd->nrrd_->axis[2].spacing = z_spacing_;
    }
    
    nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 0, nrrdCenterNode);
    nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 1, nrrdCenterNode);
    nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 2, nrrdCenterNode);
    break;
  case 2:
    if (incr) {
      return 0;
    } else {
      sciNrrd->nrrd_->axis[0].label = airStrdup( "x" );
      sciNrrd->nrrd_->axis[1].label = airStrdup( "y" );
      sciNrrd->nrrd_->axis[0].spacing = arr[0]->axis[0].spacing; 
      sciNrrd->nrrd_->axis[1].spacing = arr[0]->axis[1].spacing;

      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 0, nrrdCenterNode);
      nrrdAxisInfoMinMaxSet(sciNrrd->nrrd_, 1, nrrdCenterNode);
    }
    break;
  default:
    return 0;
  }

  return sciNrrd;
}


} // End namespace SCITeem

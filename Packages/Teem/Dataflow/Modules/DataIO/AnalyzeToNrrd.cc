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
 * C++ (CC) FILE : AnalyzeToNrrd.cc
 *
 * DESCRIPTION   : This module allows an arbitrary number of Analyze files
 *                 to be read in and converted to 4D nrrds for processing and
 *                 visualization.  The UI associated with this module provides
 *                 a mechanism for selecting Analyze files.
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *         
 * CREATED       : 9/19/2003
 * MODIFIED      : 10/4/2003
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
#include <Core/Algorithms/DataIO/AnalyzeReader.h>
#include <Core/Algorithms/DataIO/AnalyzeImage.h>
#endif

namespace SCITeem {

using namespace SCIRun;

// ****************************************************************************
// *************************** Class: AnalyzeToNrrd ***************************
// ****************************************************************************
 
class TeemSHARE AnalyzeToNrrd : public Module 
{

public:

  //! Constructor
  AnalyzeToNrrd(GuiContext*);

  //! Destructor
  virtual ~AnalyzeToNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

#ifdef HAVE_INSIGHT

  int build_nrrds( vector<Nrrd*> & array );

  NrrdData * join_nrrds( vector<Nrrd*> arr );

#endif

private:

  //! GUI variables
  GuiString file_;
  GuiString file_del_;
  GuiString messages_;

  //! Ports
  NrrdOPort*      onrrd_;

  //! Set of Analyze files 
  vector<string> all_files_;

};

DECLARE_MAKER(AnalyzeToNrrd)

/*===========================================================================*/
// 
// AnalyzeToNrrd
//
// Description : Constructor
//
// Arguments   :
//
// GuiContext* ctx - GUI context
//
AnalyzeToNrrd::AnalyzeToNrrd(GuiContext* ctx)
  : Module("AnalyzeToNrrd", ctx, Source, "DataIO", "Teem"),
    file_(ctx->subVar("file")),
    file_del_(ctx->subVar("file-del")),
    messages_(ctx->subVar("messages"))
{
}

/*===========================================================================*/
// 
// ~AnalyzeToNrrd
//
// Description : Destructor
//
// Arguments   : none
//
AnalyzeToNrrd::~AnalyzeToNrrd(){
}


/*===========================================================================*/
// 
// execute 
//
// Description : The execute function for this module.  This is the control
//               center for the module.  This reads a bunch of Analyze files,
//               constructs a nrrd with the Analyze data, and sends the nrrd
//               downstream.
//
// Arguments   : none
//
void AnalyzeToNrrd::execute(){

#ifdef HAVE_INSIGHT

  // If no Analyze files were specified via the UI, print error and return
  if( all_files_.size() == 0 ) 
  {
    error("(AnalyzeToNrrd::execute) No Analyze files specified -- use 'UI' button to select Analyze files.");
    return;
  }
  
  // Build a vector of nrrds from one or more Analyze files
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
    error("(AnalyzeToNrrd::execute) Failed to join nrrds.");
    return;
  }

  // Create handle to data
  NrrdDataHandle sciNrrdHandle(sciNrrd);

  // Initialize output port
  onrrd_ = (NrrdOPort *)get_oport("Nrrd");

  if( !onrrd_ ) {
    error("(AnalyzeToNrrd::execute) Unable to initialize oport 'Nrrd'.");
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
  error("(AnalyzeToNrrd::execute) Cannot read Analyze files.  Insight module needs to be included.");
  return;
#endif

}

/*===========================================================================*/
// 
// split_string
//
// Description : Splits a string into vector of strings based on a separator.
//
// Arguments   :
//
// const T &src - String to be split.
// C &container - Vector of strings to contain result.
// typename T::value_type splitter - Separator string.
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
void AnalyzeToNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("AnalyzeToNrrd needs a minor command");
    return;
  }

  if( args[1] == "add_data" )
  {
#ifdef HAVE_INSIGHT

    file_ = (ctx->subVar("file"));
    string file = file_.get();

    all_files_.insert( all_files_.end(), file );

#endif
  } 
  else if( args[1] == "delete_data" )
  {
#ifdef HAVE_INSIGHT

    // Get the selected file to be deleted
    file_del_ = (ctx->subVar("file-del"));
    string file_del = file_del_.get();

    // Find the matching entry in the all_files vector and remove it
    int num_files = all_files_.size();
    vector<string>::iterator iter = all_files_.begin();
     
    for( int i = 0; i < num_files; i++ )
    {
      if( file_del == all_files_[i] )
      {
        // Erase this element from the vector of files
        all_files_.erase( iter );
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
// Description : Given a set of Analyze files, reads all files in and 
//               builds one nrrd object for each file.  Returns 0 on success,
//               -1 on failure.
//
// Arguments   : 
//
// vector<Nrrd*> & array - Empty vector that will be initialized to contain
//                         one nrrd object per file.  
//
int AnalyzeToNrrd::build_nrrds( vector<Nrrd*> & array )
{

  // Get the number of files to build nrrds from 
  int num_files = all_files_.size();

  //cerr << "(AnalyzeToNrrd::build_nrrds) num_files = " << num_files << "\n";

  vector<Nrrd*> arr( num_files );
  AnalyzeReader reader;

  // Read each Analyze files and build a nrrd object from it
  for( int i = 0; i < num_files; i++ )
  {
    // Get this file
    string f = all_files_[i];

    AnalyzeImage image;
    reader.set_file( f );
    int ret;
    if( (ret = reader.read(image)) == -1 )
    {
      error("(AnalyzeToNrrd::build_nrrds) Analyze read failed.  Check command line output for details.");
      return -1;
    }

    image.print_image_info();
  
    // Construct a nrrd from the Analyze data
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
        error( "(AnalyzeToNrrd::execute) Error creating nrrd." );
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
        error( "(AnalyzeToNrrd::execute) Error creating nrrd." );
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
      error( "(AnalyzeToNrrd::build_nrrds) Invalid image dimension." );
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
NrrdData * AnalyzeToNrrd::join_nrrds( vector<Nrrd*> arr )
{
  int num_nrrds = arr.size();
  //cerr << "(AnalyzeToNrrd::join_nrrds) num_nrrds = " << num_nrrds << "\n";

  if( num_nrrds == 0 )
  {
    error( "(AnalyzeToNrrd::join_nrrds) No nrrds built." );
    return 0;
  }

  // Join all nrrds together into one 4D nrrd object
  NrrdData *sciNrrd = scinew NrrdData();
  sciNrrd->nrrd = nrrdNew();

  if( nrrdJoin(sciNrrd->nrrd, &arr[0], num_nrrds, 0, true) ) 
  {
    char *err = biffGetDone(NRRD);
    error( string("(AnalyzeToNrrd::join_nrrds) Join Error: ") +  err );
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

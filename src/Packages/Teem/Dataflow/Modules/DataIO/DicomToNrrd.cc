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
 * DESCRIPTION   : 
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *         
 * CREATED       : 9/19/2003
 *
 * MODIFIED      : 9/19/2003
 *
 * DOCUMENTATION :
 * 
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

#ifdef HAVE_INSIGHT
#include <Core/Algorithms/DataIO/DicomSeriesReader.h>
#include <Core/Algorithms/DataIO/DicomImage.h>
#endif

namespace SCITeem {

using namespace SCIRun;

// ****************************************************************************
// ***************************** Class: DicomToNrrd ***************************
// ****************************************************************************

class TeemSHARE DicomToNrrd : public Module {
public:

  //! GUI variables
  GuiString gui_filename_;

  DicomToNrrd(GuiContext*);

  virtual ~DicomToNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

#ifdef HAVE_INSIGHT

  void print_image_info( DicomImage * image );

  vector<Nrrd*> build_nrrds();

  NrrdData * join_nrrds( vector<Nrrd*> arr );

#endif

private:

  NrrdOPort*      onrrd_;

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
    gui_filename_(ctx->subVar("filename"))
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

  // Build a vector of nrrds from one or more DICOM series'
  vector<Nrrd*> arr = build_nrrds();

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
  error("(DicomToNrrd::execute) Cannot read DICOM files.  Insight module 
          needs to be included.");
  return;
#endif

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
  Module::tcl_command(args, userdata);
}

#ifdef HAVE_INSIGHT

/*===========================================================================*/
// 
// print_image_info
//
// Description : Prints image info for the single Dicom image passed.  This is
//               useful for debugging.
//
// Arguments   :
//
// DicomImage * image - Pointer to a DicomImage object that has been built
//                      from DICOM files using a DicomSeriesReader.
//
void DicomToNrrd::print_image_info( DicomImage * image )
{

  // Get data from DICOM files

  // Get number of pixels
  int num_pixels = image->get_num_pixels();
  cerr << "(DicomToNrrd::print_image_info) Num Pixels: " << num_pixels << "\n";

  // Get pixel buffer data (array)
  PixelType * pixel_data = image->get_pixel_buffer();
  //for( int i = 0; i < num_pixels; i++ )
  //{
  //  cout << "(DicomToNrrd) Pixel value " << i << ": " << pixel_data[i] 
  //       << "\n"; 
  //}

  // Get pixel type
  image->get_data_type();

  // Get image dimension
  int image_dim = image->get_dimension();
  cerr << "(DicomToNrrd::print_image_info) Dimension: " << image_dim << "\n";

  // Get the size of each axis
  cerr << "(DicomToNrrd::print_image_info) Size: [ ";
  for( int j = 0; j < image_dim; j++ )
  {
    cerr << image->get_size(j) << " "; 
  }
  cerr << "]\n";

  // Get the origin  
  cerr << "(DicomToNrrd::print_image_info) Origin: [ ";
  for( int k = 0; k < image_dim; k++ )
  {
    cerr << image->get_origin(k) << " "; 
  }
  cerr << "]\n";

  // Get the pixel spacing
  cerr << "(DicomToNrrd::print_image_info) Spacing: [ ";
  for( int m = 0; m < image_dim; m++ )
  {
    cerr << image->get_spacing(m) << " "; 
  }
  cerr << "]\n";

  // Get the indices
  cerr << "(DicomToNrrd::print_image_info) Index: [ ";
  for( int n = 0; n < image_dim; n++ )
  {
    cerr << image->get_index(n) << " "; 
  }
  cerr << "]\n";


}
 
/*===========================================================================*/
// 
// build_nrrds 
//
// Description : Given a set of DICOM series', reads all series' in and builds
//               one nrrd object for each series.  Returns a vector containing
//               the nrrds.
//
// Arguments   : none
//
vector<Nrrd*> DicomToNrrd::build_nrrds()
{

  // Read a single series of DICOM files into a DicomImage object  
  int num_series = 1;
  char * series[num_series];
  char * series0 = "/home/sci/simpson/sci_io/dicom/partial_humerus4";
  //char * series1 = "/home/sci/simpson/sci_io/dicom/partial_humerus5";
  series[0] = series0;
  //series[1] = series1;

  vector<Nrrd*> arr( num_series );
  DicomSeriesReader reader;

  // Read each DICOM series and build a nrrd object from it
  for( int i = 0; i < num_series; i++ )
  {
 
    DicomImage image = reader.read_series( series[i] );
    print_image_info( &image );
  
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

    // Add this nrrd to the vector of nrrds
    arr[i] = nrrd;
  }


  return arr;
}

/*===========================================================================*/
// 
// join_nrrds 
//
// Description : Given a vector of 3D nrrd objects, builds a single 4D nrrd
//               that is a combination of all the nrrds.  Uses the nrrdJoin
//               function to do this.
//
// Arguments   : none
//
NrrdData * DicomToNrrd::join_nrrds( vector<Nrrd*> arr )
{
  int num_nrrds = arr.size();
  cout << "(DicomToNrrd::join_nrrds) num_nrrds = " << num_nrrds << "\n";

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
    error(string("(DicomToNrrd::join_nrrds) Join Error: ") +  err);
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

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
 *  ImageToNrrd.cc:
 *
 *  Written by:
 *   Darby Van Uitert
 *   February 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include "itkImageRegionIterator.h"

#ifdef HAVE_TEEM
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>
#endif

namespace Insight {

using namespace SCIRun;
#ifdef HAVE_TEEM
using namespace SCITeem;
#endif

class PSECORESHARE ImageToNrrd : public Module {
public:
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

#ifdef HAVE_TEEM
  NrrdOPort* onrrd_;
  NrrdDataHandle onrrd_handle_;
#endif

  ImageToNrrd(GuiContext*);

  virtual ~ImageToNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class InputImageType, unsigned nrrdtype> 
  bool run( itk::Object* );  // Scalar Images

private:
  template<class InputImageType, unsigned nrrdtype>
  void create_nrrd(ITKDatatypeHandle &img);
};


DECLARE_MAKER(ImageToNrrd)

ImageToNrrd::ImageToNrrd(GuiContext* ctx)
  : Module("ImageToNrrd", ctx, Source, "Converters", "Insight")
{
}

ImageToNrrd::~ImageToNrrd(){
}

template<class InputImageType, unsigned  nrrdtype>
void ImageToNrrd::create_nrrd(ITKDatatypeHandle &img) {
  // check if Teem exists
#ifdef HAVE_TEEM

  InputImageType *im = dynamic_cast< InputImageType * >( img.get_rep()->data_.GetPointer() );
  typedef typename itk::ImageRegionIterator<InputImageType> IteratorType;

  int dim = im->GetImageDimension();

  // create a NrrdData
  NrrdData* nout = scinew NrrdData();
  nout->nrrd = nrrdNew();
  Nrrd* nr = nout->nrrd;

  // Allocate the nrrd data and add an axis of size 1 for the tuple axis.
  // Set the labels and center
  switch(dim) {
  case 1:
    nrrdAlloc(nr, nrrdtype, dim+1, 1, im->GetRequestedRegion().GetSize()[0]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "unknown:Scalar", "x");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterUnknown, nrrdCenterNode);
    break;
  case 2:
    nrrdAlloc(nr, nrrdtype, dim+1, 1, im->GetRequestedRegion().GetSize()[0],
	      im->GetRequestedRegion().GetSize()[1]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "unknown:Scalar", "x", "y");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterUnknown, nrrdCenterNode,
		    nrrdCenterNode);
    break;
  case 3:
    nrrdAlloc(nr, nrrdtype, dim+1, 1, im->GetRequestedRegion().GetSize()[0],
	      im->GetRequestedRegion().GetSize()[1],
	      im->GetRequestedRegion().GetSize()[2]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "unknown:Scalar", "x", "y", "z");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterUnknown, nrrdCenterNode,
		    nrrdCenterNode, nrrdCenterNode);
    break;
  default:
    error("Can only convert images of dimension < 4.");
    break;
  }


  // set spacing, and min and max
  // Set the spacing, and origin
  nr->axis[0].spacing = AIR_NAN;
  nr->axis[0].min = AIR_NAN;
  for(int i=1; i<=dim; i++) {
    nr->axis[i].spacing = im->GetSpacing()[i-1];

    nr->axis[i].min = im->GetOrigin()[i-1];
    nr->axis[i].max = ceil(im->GetOrigin()[i-1] + 
      ((nr->axis[i].size-1) * nr->axis[i].spacing));
    //nrrdAxisMinMaxSet(nr,i, nrrdCenterNode);
  }

  // iterate through the ITK requested region and copy the data
  IteratorType img_iter(im, im->GetRequestedRegion());
  void* p = nr->data;

  // When iterating over the nrrd to copy it
  // we need to skip the initial tuple axis.
  typedef typename InputImageType::PixelType PixelType;
  PixelType *&i = (PixelType*&)p;
  ++i;
  
  img_iter.GoToBegin();
  while(!img_iter.IsAtEnd()) {
    PixelType *&i = (PixelType*&)p;
    *i = img_iter.Get();
    
    // increment pointers
    img_iter.operator++();
    ++i;
  }

  onrrd_handle_ = nout;
#endif
  
}

template<class InputImageType, unsigned nrrdtype>
bool ImageToNrrd::run( itk::Object* obj1) 
{
  InputImageType* n = dynamic_cast< InputImageType * >(obj1);
  if( !n ) {
    return false;
  }
  
  create_nrrd<InputImageType,nrrdtype>(inhandle1_);

#ifdef HAVE_TEEM
  if(onrrd_handle_.get_rep()) {
    onrrd_->send(onrrd_handle_);
  }
#endif
  return true;
}

void ImageToNrrd::execute() {
  inport1_ = (ITKDatatypeIPort *)get_iport("InputImage");
#ifdef HAVE_TEEM
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");
#endif
  
  if (!inport1_) {
    error("Unable to initialize iport 'InputImage'.");
    return;
  }
#ifdef HAVE_TEEM
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
#endif  
  if(!inport1_->get(inhandle1_))
    return;
  
  // get input
  itk::Object *n = inhandle1_.get_rep()->data_.GetPointer();
#ifdef HAVE_TEEM  
  // can we operate on it?
  if(0) { }
  else if(run< itk::Image<float, 2>, nrrdTypeFloat >(n)) { }
  else if(run< itk::Image<float, 3>, nrrdTypeFloat >(n)) { }
  else if(run< itk::Image<double, 2>, nrrdTypeDouble >(n)) { }
  else if(run< itk::Image<double, 3>, nrrdTypeDouble >(n)) { }
  else if(run< itk::Image<int, 2>, nrrdTypeInt >(n)) { }
  else if(run< itk::Image<int, 3>, nrrdTypeInt >(n)) { }
  else if(run< itk::Image<unsigned char, 2>, nrrdTypeUChar >(n)) { }
  else if(run< itk::Image<unsigned char, 3>, nrrdTypeUChar >(n)) { }
  else if(run< itk::Image<char, 2>, nrrdTypeChar >(n)) { }
  else if(run< itk::Image<char, 3>, nrrdTypeChar >(n)) { }
  else if(run< itk::Image<unsigned short, 2>, nrrdTypeUShort >(n)) { }
  else if(run< itk::Image<unsigned short, 3>, nrrdTypeUShort >(n)) { }
  else if(run< itk::Image<short, 2>, nrrdTypeShort >(n)) { }
  else if(run< itk::Image<short, 3>, nrrdTypeShort >(n)) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }
#else
  error("Must have Teem to use this module.  Please reconfigure and enable the Teem package.");
#endif
}

void ImageToNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight



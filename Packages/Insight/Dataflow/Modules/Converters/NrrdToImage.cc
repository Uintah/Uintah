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
 *  NrrdToImage.cc:
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

class PSECORESHARE NrrdToImage : public Module {
public:
#ifdef HAVE_TEEM
  NrrdIPort *inrrd_;
  NrrdDataHandle inrrd_handle_;
#endif

  ITKDatatypeOPort *oimg_;
  ITKDatatypeHandle oimg_handle_;

  NrrdToImage(GuiContext*);

  virtual ~NrrdToImage();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // determine the nrrd pixel type and call create_image
  template<unsigned int dim>
  void determine_nrrd_type();

  // create an ITK image from the nrrd
  // stored in inrrd_handle_
  template<class type, unsigned int dim>
  void create_image();
private:
  bool remove_tuple_;
};


DECLARE_MAKER(NrrdToImage)
NrrdToImage::NrrdToImage(GuiContext* ctx)
  : Module("NrrdToImage", ctx, Source, "Converters", "Insight")
{
  remove_tuple_ = false;
}

NrrdToImage::~NrrdToImage(){
}

void NrrdToImage::execute(){
#ifdef HAVE_TEEM

  inrrd_ = (NrrdIPort*)get_iport("InputNrrd");
  if(!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }

  oimg_ = (ITKDatatypeOPort*)get_oport("OutputImage");
  if(!oimg_) {
    error("Unable to initialize oport 'OututImage'.");
    return;
  }

  if(!inrrd_->get(inrrd_handle_))
    return;

  Nrrd *n = inrrd_handle_->nrrd;
  int dim = n->dim;

  // if first axis has a spacing of nan, then
  // it is a tuple axis and we should not include it 
  // in our conversion
  remove_tuple_ = false;
  if ( !AIR_EXISTS(n->axis[0].spacing)) {
    remove_tuple_ = true;
  } 

  
  switch(dim) {
  case 1:
    if (remove_tuple_) 
      error("Cannot convert nrrd with only a tuple axis");
    else
      determine_nrrd_type<1>();
    break;
  case 2:
    if (remove_tuple_)
      determine_nrrd_type<1>();
    else
      determine_nrrd_type<2>();
    break;
  case 3:
    if (remove_tuple_)
    determine_nrrd_type<2>();
    else
      determine_nrrd_type<3>();
    break;
  case 4:
    if (remove_tuple_)
      determine_nrrd_type<3>();
    else
      error("Cannot convert nrrd with 4 dimension and tuple axis");
    break;
  default:
    error("Cannot convert > 3 dimensional data to an ITK Image");
    return;
  }
  oimg_->send(oimg_handle_);
  
#else
  error("Must have Teem to use this module. Please reconfigure and enable Teem");
  return;
#endif
}

template<unsigned int dim>
void NrrdToImage::determine_nrrd_type() {
#ifdef HAVE_TEEM

  Nrrd* n = inrrd_handle_->nrrd;

  // determine pixel type
  switch(n->type) {
  case nrrdTypeChar:
    create_image<char,dim>();
    break;
  case nrrdTypeUChar:
    create_image<unsigned char,dim>();
    break;
  case nrrdTypeShort:
    create_image<short,dim>();
    break;
  case nrrdTypeUShort:
    create_image<unsigned short,dim>();
    break;
  case nrrdTypeInt:
    create_image<int,dim>();
    break;
  case nrrdTypeUInt:
    create_image<unsigned int,dim>();
    break;
  case nrrdTypeFloat:
    create_image<float,dim>();
    break;
  case nrrdTypeDouble:
    create_image<double,dim>();
    break;
  }
#endif
}

template<class type, unsigned int dim>
void NrrdToImage::create_image() {
#ifdef HAVE_TEEM

  Nrrd* n = inrrd_handle_->nrrd;
  typedef typename itk::Image<type,dim> ImageType;
  typedef typename itk::ImageRegionIterator< ImageType > IteratorType;

  // create new itk image
  typename ImageType::Pointer img = ImageType::New();

  typename ImageType::RegionType region;

  // set size, origin and spacing
  typename ImageType::SizeType fixedSize;
  typename ImageType::IndexType start;
  double origin[dim];
  double spacing[dim];

  // If tuple axis is being removed, we need to offset indexing
  // into the nrrd by 1.
  int offset = 0;
  if (remove_tuple_)
    offset = 1;

  for(int i=0; i<(int)dim; i++) {
    fixedSize[i] = n->axis[i+offset].size;

    start[i] = 0;

    if (!AIR_EXISTS(n->axis[i+offset].min))
      origin[i] = 0;
    else
    origin[i] = n->axis[i+offset].min;

    if (!AIR_EXISTS(n->axis[i+offset].spacing))
      spacing[i] = 1.0;
    else
      spacing[i] = n->axis[i+offset].spacing;
  }
  region.SetSize( fixedSize );
  region.SetIndex( start );
  
  img->SetRegions( region );
  img->Allocate();

  img->SetOrigin( origin );
  img->SetSpacing( spacing );

  // copy the data
  IteratorType img_iter(img, img->GetRequestedRegion());
  void* p = n->data;

  // If remove_tuple_ is true, then the dimension of the nrrd
  // is really dim+1.  When iterating over the nrrd to copy it
  // we need to skip the initial tuple axis.
  if (remove_tuple_) {
    for(int j=0; j<n->axis[0].size; j++) {
      type *&i = (type*&)p;
      // increment pointer
      ++i;
    }
  }
  
  img_iter.GoToBegin();
  while(!img_iter.IsAtEnd()) {
    type *&i = (type*&)p;
    type v = *i;
    img_iter.Set(v);
    
    // increment pointers
    img_iter.operator++();
    ++i;
  }

  ITKDatatype* result = scinew ITKDatatype;
  result->data_ = img;
  oimg_handle_ = result;

#endif
}

void NrrdToImage::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight



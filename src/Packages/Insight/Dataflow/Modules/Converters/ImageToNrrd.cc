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
 *  ImageToNrrd.cc:
 *
 *  Written by:
 *   Darby Van Uitert
 *   February 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include "itkImageRegionIterator.h"
#include <itkVector.h>

#include <sci_defs/teem_defs.h>

#include <Dataflow/Ports/NrrdPort.h>

namespace Insight {

using namespace SCIRun;

class ImageToNrrd : public Module {
public:
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;

  NrrdOPort* onrrd_;
  NrrdDataHandle onrrd_handle_;

  ImageToNrrd(GuiContext*);

  virtual ~ImageToNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class InputImageType, unsigned nrrdtype> 
  bool run( itk::Object* );  // Scalar Images
  template< class InputImageType, unsigned nrrdtype> 
  bool run2( itk::Object* );  // Vector Images

private:
  template<class InputImageType, unsigned nrrdtype>
  void create_nrrd(ITKDatatypeHandle &img);
  template<class InputImageType, unsigned nrrdtype>
  void create_nrrd2(ITKDatatypeHandle &img);
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
    nrrdAlloc(nr, nrrdtype, dim, im->GetRequestedRegion().GetSize()[0]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "x");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterNode);
    break;
  case 2:
    nrrdAlloc(nr, nrrdtype, dim, im->GetRequestedRegion().GetSize()[0],
	      im->GetRequestedRegion().GetSize()[1]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "x", "y");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterNode,  nrrdCenterNode);
    break;
  case 3:
    nrrdAlloc(nr, nrrdtype, dim, im->GetRequestedRegion().GetSize()[0],
	      im->GetRequestedRegion().GetSize()[1],
	      im->GetRequestedRegion().GetSize()[2]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "x", "y", "z");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterNode,
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
  for(int i=0; i<dim; i++) {
    nr->axis[i].spacing = im->GetSpacing()[i];

    nr->axis[i].min = im->GetOrigin()[i];
    nr->axis[i].max = ceil(im->GetOrigin()[i] + 
      ((nr->axis[i].size-1) * nr->axis[i].spacing));
    //nrrdAxisMinMaxSet(nr,i, nrrdCenterNode);
  }

  // iterate through the ITK requested region and copy the data
  IteratorType img_iter(im, im->GetRequestedRegion());
  void* p = nr->data;


  typedef typename InputImageType::PixelType PixelType;
  PixelType *&i = (PixelType*&)p;
  
  img_iter.GoToBegin();
  while(!img_iter.IsAtEnd()) {
    //PixelType *&i = (PixelType*&)p;
    *i = img_iter.Get();
    
    // increment pointers
    img_iter.operator++();
    ++i;
  }

  onrrd_handle_ = nout;
  
}

template<class InputImageType, unsigned  nrrdtype>
void ImageToNrrd::create_nrrd2(ITKDatatypeHandle &img) {
  // check if Teem exists

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
    nrrdAlloc(nr, nrrdtype, dim+1, 3, im->GetRequestedRegion().GetSize()[0]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "rgb", "x");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterUnknown, nrrdCenterNode);
    break;
  case 2:
    nrrdAlloc(nr, nrrdtype, dim+1, 3, im->GetRequestedRegion().GetSize()[0],
	      im->GetRequestedRegion().GetSize()[1]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "rgb", "x", "y");
    nrrdAxisInfoSet(nr, nrrdAxisInfoCenter, nrrdCenterUnknown, nrrdCenterNode,  
		    nrrdCenterNode);
    break;
  case 3:
    nrrdAlloc(nr, nrrdtype, dim+1, 3, im->GetRequestedRegion().GetSize()[0],
	      im->GetRequestedRegion().GetSize()[1],
	      im->GetRequestedRegion().GetSize()[2]);
    nrrdAxisInfoSet(nr, nrrdAxisInfoLabel, "rgb", "x", "y", "z");
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
  nr->axis[0].kind = nrrdKind3Vector;

  for(int i=0; i<dim; i++) {
    nr->axis[i+1].spacing = im->GetSpacing()[i];

    nr->axis[i+1].min = im->GetOrigin()[i];
    nr->axis[i+1].max = ceil(im->GetOrigin()[i] + 
      ((nr->axis[i+1].size-1) * nr->axis[i+1].spacing));
    nr->axis[i+1].kind = nrrdKindDomain;
  }

  // iterate through the ITK requested region and copy the data
  IteratorType img_iter(im, im->GetRequestedRegion());
  void* p = nr->data;

  typedef typename InputImageType::PixelType::ValueType ValueType;
  ValueType *&i = (ValueType*&)p;

  img_iter.GoToBegin();
  //int count = 0;
  while(!img_iter.IsAtEnd()) {
    //ValueType *&i = (ValueType*&)p;
    *i = img_iter.Get()[0];
    ++i;
    *i = img_iter.Get()[1];
    ++i;
    *i = img_iter.Get()[2];
    ++i;
    
    // increment pointers
    img_iter.operator++();
  }

  onrrd_handle_ = nout;
}

template<class InputImageType, unsigned nrrdtype>
bool ImageToNrrd::run( itk::Object* obj1) 
{
  InputImageType* n = dynamic_cast< InputImageType * >(obj1);
  if( !n ) {
    return false;
  }
  
  create_nrrd<InputImageType,nrrdtype>(inhandle1_);

  if(onrrd_handle_.get_rep()) {
    onrrd_->send(onrrd_handle_);
  }
  return true;
}

template<class InputImageType, unsigned nrrdtype>
bool ImageToNrrd::run2( itk::Object* obj1) 
{
  cerr << "run2\n";
  InputImageType* n = dynamic_cast< InputImageType * >(obj1);
  if( !n ) {
    return false;
  }
  
  create_nrrd2<InputImageType,nrrdtype>(inhandle1_);

  if(onrrd_handle_.get_rep()) {
    onrrd_->send(onrrd_handle_);
  }
  return true;
}

void ImageToNrrd::execute() {
  inport1_ = (ITKDatatypeIPort *)get_iport("InputImage");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");  
  
  if (!inport1_) {
    error("Unable to initialize iport 'InputImage'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }

  inport1_->get(inhandle1_);
  
  if(!inhandle1_.get_rep())
    return;
  
  // get input
  itk::Object *n = inhandle1_.get_rep()->data_.GetPointer();
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
  else if(run2< itk::Image<itk::Vector<float>, 2 >, nrrdTypeFloat >(n)) { }
  else if(run2< itk::Image<itk::Vector<float>, 3 >, nrrdTypeFloat >(n)) { }
  else if(run2< itk::Image<itk::Vector<double>, 2 >, nrrdTypeDouble >(n)) { }
  else if(run2< itk::Image<itk::Vector<double>, 3 >, nrrdTypeDouble >(n)) { }
  else if(run2< itk::Image<itk::Vector<char>, 2 >, nrrdTypeChar >(n)) { }
  else if(run2< itk::Image<itk::Vector<char>, 3 >, nrrdTypeChar >(n)) { }
  else if(run2< itk::Image<itk::Vector<unsigned char>, 2 >, nrrdTypeUChar >(n)) { }
  else if(run2< itk::Image<itk::Vector<unsigned char>, 3 >, nrrdTypeUChar >(n)) { }
  else if(run2< itk::Image<itk::Vector<short>, 2 >, nrrdTypeShort >(n)) { }
  else if(run2< itk::Image<itk::Vector<short>, 3 >, nrrdTypeShort >(n)) { }
  else if(run2< itk::Image<itk::Vector<unsigned short>, 2 >, nrrdTypeUShort >(n)) { }
  else if(run2< itk::Image<itk::Vector<unsigned short>, 3 >, nrrdTypeUShort >(n)) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }
}

void ImageToNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight



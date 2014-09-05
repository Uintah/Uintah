/* 
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Dataflow/Network/Ports/ITKDatatypePort.h>
#include "itkImageRegionIterator.h"
#include "itkMetaDataObject.h"

#include <sci_defs/teem_defs.h>

#include <Dataflow/Network/Ports/NrrdPort.h>

namespace Insight {

using namespace SCIRun;

class NrrdToImage : public Module {
public:
  NrrdDataHandle inrrd_handle_;

  ITKDatatypeHandle oimg_handle_;

  NrrdToImage(GuiContext*);

  virtual ~NrrdToImage();

  virtual void execute();

  // determine the nrrd pixel type and call create_image
  template<unsigned int dim>
  void determine_nrrd_type();

  // create an ITK image from the nrrd
  // stored in inrrd_handle_
  template<class type, unsigned int dim>
  void create_image();
  template<class type, unsigned int dim>
  void create_image2();


  void copy_kvp_to_dictionary(Nrrd *, itk::Object *);

private:
  bool vector_data_;
};


DECLARE_MAKER(NrrdToImage)
NrrdToImage::NrrdToImage(GuiContext* ctx)
  : Module("NrrdToImage", ctx, Source, "Converters", "Insight")
{
}


NrrdToImage::~NrrdToImage()
{
}


void
NrrdToImage::execute()
{
  if (!get_input_handle("InputNrrd", inrrd_handle_)) return;

  Nrrd *n = inrrd_handle_->nrrd_;
  int dim = n->dim;

  vector_data_ = false;
  if (nrrdKindSize(n->axis[0].kind) == 3) {
    remark("Vector data");
    vector_data_ = true;
  }
  
  switch(dim) {
  case 1:
    if (vector_data_) 
      error("Cannot convert nrrd with only vector data");
    else
      determine_nrrd_type<1>();
    break;
  case 2:
    if (vector_data_)
      determine_nrrd_type<1>();
    else
      determine_nrrd_type<2>();
    break;
  case 3:
    if (vector_data_)
    determine_nrrd_type<2>();
    else
      determine_nrrd_type<3>();
    break;
  case 4:
    if (vector_data_)
      determine_nrrd_type<3>();
    else
      error("Cannot convert nrrd with 4 dimension and vector data");
    break;
  default:
    error("Cannot convert > 3 dimensional data to an ITK Image");
    return;
  }

  send_output_handle("OutputImage", oimg_handle_, true);
  
  return;
}


template<unsigned int dim>
void
NrrdToImage::determine_nrrd_type()
{
  Nrrd* n = inrrd_handle_->nrrd_;

  // determine pixel type
  switch(n->type) {
  case nrrdTypeChar:
    if (vector_data_)
      create_image2<char,dim>();
    else
      create_image<char,dim>();
    break;
  case nrrdTypeUChar:
    if (vector_data_)
      create_image2<unsigned char,dim>();
    else 
      create_image<unsigned char,dim>();
    break;
  case nrrdTypeShort:
    if (vector_data_)
      create_image2<short,dim>();
    else
      create_image<short,dim>();
    break;
  case nrrdTypeUShort:
    if (vector_data_)
      create_image2<unsigned short,dim>();
    else
      create_image<unsigned short,dim>();
    break;
  case nrrdTypeInt:
    if (vector_data_)
      create_image2<int,dim>();
    else
      create_image<int,dim>();
    break;
  case nrrdTypeUInt:
    if (vector_data_)
      create_image2<unsigned int,dim>();
    else
      create_image<unsigned int,dim>();
    break;
  case nrrdTypeFloat:
    if (vector_data_)
      create_image2<float,dim>();
    else
      create_image<float,dim>();
    break;
  case nrrdTypeDouble:
    if (vector_data_)
      create_image2<double,dim>();
    else
      create_image<double,dim>();
    break;
  }
}


template<class type, unsigned int dim>
void
NrrdToImage::create_image()
{
  Nrrd* n = inrrd_handle_->nrrd_;
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

  for(int i=0; i<(int)dim; i++) {
    fixedSize[i] = n->axis[i].size;

    start[i] = 0;

    if (!AIR_EXISTS(n->axis[i].min))
      origin[i] = 0;
    else
    origin[i] = n->axis[i].min;

    if (!AIR_EXISTS(n->axis[i].spacing))
      spacing[i] = 1.0;
    else
      spacing[i] = n->axis[i].spacing;
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

  img_iter.GoToBegin();
  while(!img_iter.IsAtEnd()) {
    type *&i = (type*&)p;
    type v = *i;
    img_iter.Set(v);
    
    // increment pointers
    img_iter.operator++();
    ++i;
  }

  copy_kvp_to_dictionary(n, img);

  
  ITKDatatype* result = scinew ITKDatatype;
  result->data_ = img;
  oimg_handle_ = result;

}


void
NrrdToImage::copy_kvp_to_dictionary(Nrrd *n, itk::Object *obj)
{
  // handle any key/value pairs
  itk::MetaDataDictionary &dic = obj->GetMetaDataDictionary();
  for (unsigned int i=0; i< nrrdKeyValueSize(n); i++) {
    char *key, *val;
    nrrdKeyValueIndex(n, &key, &val, i);
    if (key && val) { // add the key/value pair to the dictionary
      //      cerr << get_id() << ": " << key << ", " << val <<std::endl;
      itk::EncapsulateMetaData<string>(dic, string(key), string(val));
    }
  }
}


template<class type, unsigned int dim>
void
NrrdToImage::create_image2()
{
  Nrrd* n = inrrd_handle_->nrrd_;
  typedef typename itk::Image<itk::Vector<type>,dim> ImageType;
  typedef typename itk::ImageRegionIterator< ImageType > IteratorType;

  // create new itk image
  typename ImageType::Pointer img = ImageType::New();

  typename ImageType::RegionType region;

  // set size, origin and spacing
  typename ImageType::SizeType fixedSize;
  typename ImageType::IndexType start;
  double origin[dim];
  double spacing[dim];

  // If vector data, we need to offset indexing
  // into the nrrd by 1.
  int offset = 0;
  if (vector_data_)
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

  img_iter.GoToBegin();
  while(!img_iter.IsAtEnd()) {
    type *&i = (type*&)p;
    type x = *i;
    ++i;
    type y = *i;
    ++i;
    type z = *i;  
    ++i;

    itk::Vector<type> temp;
    temp[0] = x;
    temp[1] = y;
    temp[2] = z;
    img_iter.Set(temp);
    
    // increment pointers
    img_iter.operator++();
  }

  ITKDatatype* result = scinew ITKDatatype;
  result->data_ = img;
  oimg_handle_ = result;

}

} // End namespace Insight



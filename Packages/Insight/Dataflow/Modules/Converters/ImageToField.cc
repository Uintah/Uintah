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
 *  ImageToField.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/ImageField.h>
#include <Packages/Insight/Core/Datatypes/ITKLatVolField.h>

#include <Core/Datatypes/ImageMesh.h>

#include <Packages/Insight/share/share.h>

namespace Insight {

using namespace SCIRun;

class InsightSHARE ImageToField : public Module {  
public:
  ITKDatatypeIPort* inrrd;
  ITKDatatypeHandle ninH;

  FieldOPort* ofield;
  FieldHandle ofield_handle;

public:
  ImageToField(GuiContext*);

  virtual ~ImageToField();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class InputImageType > 
  bool run( itk::Object* );

private:
  template<class InputImageType>
  FieldHandle create_image_field(ITKDatatypeHandle &im);

  template<class InputImageType>
  FieldHandle create_latvol_field(ITKDatatypeHandle &im);
  
};


DECLARE_MAKER(ImageToField)
ImageToField::ImageToField(GuiContext* ctx)
  : Module("ImageToField", ctx, Source, "Converters", "Insight")
{
}

ImageToField::~ImageToField(){
}

template<class InputImageType>
FieldHandle ImageToField::create_image_field(ITKDatatypeHandle &nrd) {

  typedef ImageField<typename InputImageType::PixelType> ImageFieldType;
  InputImageType *n = dynamic_cast< InputImageType * >( nrd.get_rep()->data_.GetPointer() );

  double spc[2];
  double data_center = n->GetOrigin()[0];
  
  unsigned int size_x = (n->GetLargestPossibleRegion()).GetSize()[0];
  unsigned int size_y = (n->GetLargestPossibleRegion()).GetSize()[1];

  Point min(0., 0., 0.);
  Point max(size_x, size_y, 0.);

  //ImageMesh* m = new ImageMesh(size_x+1, size_y+1, min, max);
  ImageMesh* m = new ImageMesh(size_x, size_y, min, max);

  ImageMeshHandle mh(m);

  FieldHandle fh;
  int mn_idx, mx_idx;
  
  // assume data type is unsigned char
  fh = new ImageFieldType(mh, Field::NODE); 
  ImageMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);

  // fill data
  typename InputImageType::IndexType pixelIndex;
  typedef typename ImageFieldType::value_type val_t;
  val_t tmp;
  ImageFieldType* fld = (ImageFieldType* )fh.get_rep();

  
  for(int row=0; row < size_y; row++) {
    for(int col=0; col < size_x; col++) {
      if(iter == end) {
	return fh;
      }
      pixelIndex[0] = col;
      pixelIndex[1] = row;

      tmp = n->GetPixel(pixelIndex);
      fld->set_value(tmp, *iter);
      ++iter;
    }
  }

  return fh;
}

template<class InputImageType>
FieldHandle ImageToField::create_latvol_field(ITKDatatypeHandle &nrd) {
 
  typedef ITKLatVolField<typename InputImageType::PixelType> LatVolFieldType;
  InputImageType *n = dynamic_cast< InputImageType * >( nrd.get_rep()->data_.GetPointer() );

  double spc[2];
  double data_center = n->GetOrigin()[0];
  
  unsigned int size_x = (n->GetRequestedRegion()).GetSize()[0];
  unsigned int size_y = (n->GetRequestedRegion()).GetSize()[1];
  unsigned int size_z = (n->GetRequestedRegion()).GetSize()[2];

  Point min(0., 0., 0.);
  Point max(size_x, size_y, size_z);

  LatVolMesh* m = new LatVolMesh(size_x, size_y, size_z, min, max);

  LatVolMeshHandle mh(m);

  FieldHandle fh;
  int mn_idx, mx_idx;
  
  fh = new LatVolFieldType(mh, Field::NODE, n); 
  // LatVolFieldType* fld = (LatVolFieldType* )fh.get_rep();

  return fh;
}

template<class InputImageType >
bool ImageToField::run( itk::Object* obj1) 
{
   InputImageType* n = dynamic_cast< InputImageType * >(obj1);
  if( !n ) {
    return false;
  }

  bool dim_based_convert = true;
  
  // do a standard dimension based convert
  if(dim_based_convert) {
    int dim = n->GetImageDimension();
    
    switch(dim) {
      
    case 2:
      ofield_handle = create_image_field<InputImageType>(ninH);
      break;
      
    case 3:
      ofield_handle = create_latvol_field<InputImageType>(ninH);
      break;
    default:
      error("Cannot convert data that is not 2D or 3D to a SCIRun Field.");
      return false;
    }
  }
  return true;
}

void ImageToField::execute(){
  inrrd = (ITKDatatypeIPort *)get_iport("InputImage");
  ofield = (FieldOPort *)get_oport("OutputImage");

  if (!inrrd) {
    error("Unable to initialize iport 'InputImage'.");
    return;
  }
  if (!ofield) {
    error("Unable to initialize oport 'OutputImage'.");
    return;
  }

  if(!inrrd->get(ninH))
    return;

  // get input
  itk::Object *n = ninH.get_rep()->data_.GetPointer();

  // can we operate on it?
  if(0) { }
  else if(run< itk::Image<float, 2> >(n)) { }
  else if(run< itk::Image<float, 3> >(n)) { }
  else if(run< itk::Image<unsigned char, 2> >(n)) { }
  else if(run< itk::Image<unsigned short, 2> >(n)) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }
  ofield->send(ofield_handle);
}

void ImageToField::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}



} // End namespace Insight



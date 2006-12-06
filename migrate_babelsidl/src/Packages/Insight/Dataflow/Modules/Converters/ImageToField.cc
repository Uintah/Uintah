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
 *  ImageToField.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>


#include <Core/Util/TypeDescription.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>

// Making sure classes are defined in an order so that the compiler will not fail.

namespace Insight {

template <class Data> class ITKFData3d;
template <class Data> class ITKFData2d;
template <class Data> class ITKImageField;
template <class Data> class ITKLatVolField;

}

// gotta put here (instead of in one of the .h files that also declare them)
// to satisfy templating issues in gcc3.4+
#undef SCISHARE
#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif

namespace SCIRun {
  
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<SCIRun::Tensor>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<SCIRun::Vector>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<double>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<float>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<int>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<long>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<short>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<char>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned int>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned short>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned char>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData3d<unsigned long>*);

  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<SCIRun::Tensor>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<SCIRun::Vector>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<double>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<float>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<int>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<long>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<short>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<char>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<unsigned int>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<unsigned short>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<unsigned char>*);
  SCISHARE const TypeDescription* get_type_description(Insight::ITKFData2d<unsigned long>*);

}

#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>


#include <Packages/Insight/Core/Datatypes/ITKImageField.h>
#include <Packages/Insight/Core/Datatypes/ITKLatVolField.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <Core/Datatypes/GenericField.h>


#include "itkVector.h"
#include "itkRGBPixel.h"

namespace Insight {
  
using namespace SCIRun;
  
class ImageToField : public Module {  

typedef LatVolMesh<HexTrilinearLgn<Point> >         LVMesh;
typedef ImageMesh<QuadBilinearLgn<Point> >          IMesh;

public:
  ITKDatatypeIPort* inport1_;
  ITKDatatypeHandle inhandle1_;
  
  FieldOPort* ofield_;
  FieldHandle ofield_handle_;
  
  GuiInt gui_copy_;
  
public:
  ImageToField(GuiContext*);
  
  virtual ~ImageToField();
  
  virtual void execute();
  
  virtual void tcl_command(GuiArgs&, void*);
  
  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class InputImageType > 
  bool run( itk::Object* );  // Scalar Images
  
  template< class Type, unsigned int length, unsigned int dimension > 
  bool run2( itk::Object* ); // Vector Images
  template< class Type, unsigned int dimension > 
  bool run3( itk::Object* ); // RGBPixel Images
  
private:
  template<class InputImageType>
  FieldHandle create_image_field(ITKDatatypeHandle &img);
  
  template<class InputImageType>
  FieldHandle create_latvol_field(ITKDatatypeHandle &img);
  
  template<class Type, unsigned int length>
  FieldHandle create_latvol_vector_field1(ITKDatatypeHandle &img);
  
  template<class Type, unsigned int length>
  FieldHandle create_image_vector_field1(ITKDatatypeHandle &img);

  template<class Type>
  FieldHandle create_latvol_vector_field2(ITKDatatypeHandle &img);
  
  template<class Type>
  FieldHandle create_image_vector_field2(ITKDatatypeHandle &img);
};
  
  
DECLARE_MAKER(ImageToField)
ImageToField::ImageToField(GuiContext* ctx)
  : Module("ImageToField", ctx, Source, "Converters", "Insight"),
    gui_copy_(get_ctx()->subVar("copy"))
{
}
  
ImageToField::~ImageToField(){
}
  
template<class InputImageType>
FieldHandle ImageToField::create_image_field(ITKDatatypeHandle &img) {
  InputImageType *n = dynamic_cast< InputImageType * >( img.get_rep()->data_.GetPointer() );
  typedef typename InputImageType::PixelType pType;
  typedef ITKImageField<pType> ITKIF;
  typedef GenericField<IMesh, QuadBilinearLgn<pType>, FData2d<pType, IMesh> > IF;
  
  double origin_x = n->GetOrigin()[0];
  double origin_y = n->GetOrigin()[1];
  
  // get number of data points
  unsigned int size_x = (n->GetLargestPossibleRegion()).GetSize()[0];
  unsigned int size_y = (n->GetLargestPossibleRegion()).GetSize()[1];
  
  // get spacing between data points
  double space_x = n->GetSpacing()[0];
  double  space_y = n->GetSpacing()[1];
  
  // the origin specified by the itk image should remain the same
  // so we must make the min and max points accordingly
  double spread_x = (space_x * (size_x-1));
  double spread_y = (space_y * (size_y-1));
  
  Point min(origin_x, origin_y, 0.0);
  Point max(origin_x + spread_x, origin_y + spread_y, 0.0);
  
  IMesh* m = scinew IMesh(size_x, size_y, min, max);
  
  typename IF::mesh_handle_type mh(m);
  
  FieldHandle fh;
  
  if(gui_copy_.get() == 0) {
    // simply point to the itk image
    fh = scinew ITKIF(mh, n); 
  }
  else if(gui_copy_.get() == 1) {
    // copy the data into a SCIRun ImageField
    fh = scinew IF(mh);
    IMesh::Node::iterator iter, end;
    mh->begin(iter);
    mh->end(end);
    
    // fill data
    typename InputImageType::IndexType pixelIndex;
    typedef typename IF::value_type val_t;
    val_t tmp;
    IF* fld = (IF* )fh.get_rep();
    
    for(int row=0; row < (int)size_y; row++) {
      for(int col=0; col < (int)size_x; col++) {
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
  }
  else {
    ASSERTFAIL("ImageToField Error");
  }
  return fh;
}
  
template<class InputImageType>
FieldHandle ImageToField::create_latvol_field(ITKDatatypeHandle &img) {
  
  typedef typename InputImageType::PixelType pType;
  typedef ITKLatVolField<pType> ITKLVF;
  typedef GenericField<LVMesh, HexTrilinearLgn<pType>, FData3d<pType, LVMesh> > LVF;
  
  InputImageType *n = dynamic_cast< InputImageType * >( img.get_rep()->data_.GetPointer() );
  
  // get number of data points
  unsigned int size_x = (n->GetRequestedRegion()).GetSize()[0];
  unsigned int size_y = (n->GetRequestedRegion()).GetSize()[1];
  unsigned int size_z = (n->GetRequestedRegion()).GetSize()[2];
  
  // get spacing between data points
  float space_x = n->GetSpacing()[0];
  float space_y = n->GetSpacing()[1];
  float space_z = n->GetSpacing()[2];

  // get origin in physical space
  double origin_x = n->GetOrigin()[0];
  double origin_y = n->GetOrigin()[1];
  double origin_z = n->GetOrigin()[2];
  
  // the origin specified by the itk image should remain the same
  // so we must make the min and max points accordingly
  
  double spread_x = (space_x * (size_x-1));
  double spread_y = (space_y * (size_y-1));
  double spread_z = (space_z * (size_z-1));
  
  Point min(origin_x, origin_y, origin_z);
  Point max(origin_x + spread_x, origin_y + spread_y, origin_z + spread_z);
  
  LVMesh* m = scinew LVMesh(size_x, size_y, size_z, min, max);
  
  typename LVF::mesh_handle_type mh(m);
  
  FieldHandle fh;
  
  if(gui_copy_.get() == 0) {
    // simply reference itk image
    fh = scinew ITKLVF(mh, n);
  }
  else if(gui_copy_.get() == 1) {
    // copy the data into a SCIRun LatVolField
    fh = scinew LVF(mh); 
    LVMesh::Node::iterator iter, end;
    mh->begin(iter);
    mh->end(end);
    
    // fill data
    typename InputImageType::IndexType pixelIndex;
    typedef typename LVF::value_type val_t;
    val_t tmp;
    LVF* fld = (LVF* )fh.get_rep();
    
    for(int z=0; z < (int)size_z; z++) {
      for(int row=0; row < (int)size_y; row++) {
	for(int col=0; col < (int)size_x; col++) {
	  if(iter == end) {
	    return fh;
	  }
	  pixelIndex[0] = col;
	  pixelIndex[1] = row;
	  pixelIndex[2] = z;
	  
	  tmp = n->GetPixel(pixelIndex);
	  fld->set_value(tmp, *iter);
	  ++iter;
	}
      }
    }
  }
  else {
    ASSERTFAIL("ImageToField Error");
  }
  return fh;
}



template<class Type, unsigned int length>
FieldHandle ImageToField::create_image_vector_field1(ITKDatatypeHandle &img){
  typedef GenericField<IMesh, QuadBilinearLgn<Vector>, FData2d<Vector, IMesh> > IF;
  typedef itk::Image<itk::Vector<Type, length>,2> InputImageType;
  
  InputImageType *n = dynamic_cast<InputImageType *>( img.get_rep()->data_.GetPointer());

  
  double origin_x = n->GetOrigin()[0];
  double origin_y = n->GetOrigin()[1];
  
  // get number of data points
  unsigned int size_x = (n->GetLargestPossibleRegion()).GetSize()[0];
  unsigned int size_y = (n->GetLargestPossibleRegion()).GetSize()[1];
  
  // get spacing between data points
  double space_x = n->GetSpacing()[0];
  double  space_y = n->GetSpacing()[1];
  
  // the origin specified by the itk image should remain the same
  // so we must make the min and max points accordingly
  double spread_x = (space_x * (size_x-1));
  double spread_y = (space_y * (size_y-1));
  
  Point min(origin_x, origin_y, 0.0);
  Point max(origin_x + spread_x, origin_y + spread_y, 0.0);
  
  IMesh* m = scinew IMesh(size_x, size_y, min, max);
  
  IF::mesh_handle_type mh(m);
  
  FieldHandle fh;
  
  if(gui_copy_.get() == 0) {
    remark("Cannot reference vector data. Copying instead.");
    gui_copy_.set(1);
  }
  
  // copy the data into a SCIRun ImageField
  fh = scinew IF(mh);
  IMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);
  
  // fill data
  typename InputImageType::IndexType pixelIndex;
  typedef typename InputImageType::PixelType PixelType;
  PixelType pixel;
  IF* fld = (IF* )fh.get_rep();
  
  for(int row=0; row < (int)size_y; row++) {
    for(int col=0; col < (int)size_x; col++) {
      if(iter == end) {
	return fh;
      }
      pixelIndex[0] = col;
      pixelIndex[1] = row;
      
      pixel = n->GetPixel(pixelIndex);
      if (length == 3) {
	fld->set_value(Vector(pixel[0],pixel[1],pixel[2]), *iter);
      } else {
	error("ImageToField cannot convert vectors whose length is not equal to 3");
      }
      ++iter;
    }
  }
  return fh;
}



template<class Type, unsigned int length>
FieldHandle ImageToField::create_latvol_vector_field1(ITKDatatypeHandle &img){
  typedef GenericField<LVMesh, HexTrilinearLgn<Vector>, FData3d<Vector, LVMesh> > LVF;
  typedef itk::Image<itk::Vector<Type, length>,3> InputImageType;
  
  InputImageType *n = dynamic_cast< InputImageType * >( img.get_rep()->data_.GetPointer() );
  
  // get number of data points
  unsigned int size_x = (n->GetRequestedRegion()).GetSize()[0];
  unsigned int size_y = (n->GetRequestedRegion()).GetSize()[1];
  unsigned int size_z = (n->GetRequestedRegion()).GetSize()[2];
  
  // get spacing between data points
  float space_x = n->GetSpacing()[0];
  float space_y = n->GetSpacing()[1];
  float space_z = n->GetSpacing()[2];
  
  // get origin in physical space
  double origin_x = n->GetOrigin()[0];
  double origin_y = n->GetOrigin()[1];
  double origin_z = n->GetOrigin()[2];
  
  // the origin specified by the itk image should remain the same
  // so we must make the min and max points accordingly
  
  double spread_x = (space_x * (size_x-1));
  double spread_y = (space_y * (size_y-1));
  double spread_z = (space_z * (size_z-1));
  
  Point min(origin_x, origin_y, origin_z);
  Point max(origin_x + spread_x, origin_y + spread_y, origin_z + spread_z);
  
  LVMesh* m = scinew LVMesh(size_x, size_y, size_z, min, max);
  
  LVF::mesh_handle_type mh(m);
  
  FieldHandle fh;
  
  if(gui_copy_.get() == 0) {
    remark("Cannot reference rgb data. Copying instead.");
    gui_copy_.set(1);
  }
  
  // copy the data into a SCIRun LatVolField
  fh = scinew LVF(mh); 
  LVMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);
  
  // fill data
  typename InputImageType::IndexType pixelIndex;
  typedef typename InputImageType::PixelType PixelType;
  PixelType pixel;
  LVF* fld = (LVF* )fh.get_rep();
  
  for(int z=0; z < (int)size_z; z++) {
    for(int row=0; row < (int)size_y; row++) {
      for(int col=0; col < (int)size_x; col++) {
	if(iter == end) {
	  return fh;
	}
	pixelIndex[0] = col;
	pixelIndex[1] = row;
	pixelIndex[2] = z;
	
	pixel = n->GetPixel(pixelIndex);
	if (length == 3) {
	  fld->set_value(Vector(pixel[0], pixel[1], pixel[2]), *iter);
	} else {
	  error("ImageToField cannot convert vectors whose length is not equal to 3");
	  return 0;
	}
	++iter;
	
      }
    }
  }
  
  return fh;
}




template<class Type>
FieldHandle ImageToField::create_image_vector_field2(ITKDatatypeHandle &img){

  typedef GenericField<IMesh, QuadBilinearLgn<Vector>, FData2d<Vector, IMesh> > IF;
  typedef itk::Image<itk::RGBPixel<Type>,2> InputImageType;
  
  InputImageType *n = dynamic_cast<InputImageType *>( img.get_rep()->data_.GetPointer());

  
  double origin_x = n->GetOrigin()[0];
  double origin_y = n->GetOrigin()[1];
  
  // get number of data points
  unsigned int size_x = (n->GetLargestPossibleRegion()).GetSize()[0];
  unsigned int size_y = (n->GetLargestPossibleRegion()).GetSize()[1];
  
  // get spacing between data points
  double space_x = n->GetSpacing()[0];
  double  space_y = n->GetSpacing()[1];
  
  // the origin specified by the itk image should remain the same
  // so we must make the min and max points accordingly
  double spread_x = (space_x * (size_x-1));
  double spread_y = (space_y * (size_y-1));
  
  Point min(origin_x, origin_y, 0.0);
  Point max(origin_x + spread_x, origin_y + spread_y, 0.0);
  
  IMesh* m = scinew IMesh(size_x, size_y, min, max);
  
  IF::mesh_handle_type mh(m);
  
  FieldHandle fh;
  
  if(gui_copy_.get() == 0) {
    remark("Cannot reference vector data. Copying instead.");
    gui_copy_.set(1);
  }
  
  // copy the data into a SCIRun ImageField
  fh = scinew IF(mh);
  IMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);
  
  // fill data
  typename InputImageType::IndexType pixelIndex;
  typedef typename InputImageType::PixelType PixelType;
  PixelType pixel;
  IF* fld = (IF* )fh.get_rep();
  
  for(int row=0; row < (int)size_y; row++) {
    for(int col=0; col < (int)size_x; col++) {
      if(iter == end) {
	return fh;
      }
      pixelIndex[0] = col;
      pixelIndex[1] = row;
      
      pixel = n->GetPixel(pixelIndex);

      fld->set_value(Vector(pixel[0], pixel[1], pixel[2]), *iter);
      ++iter;
    }
  }
  return fh;
}



template<class Type>
FieldHandle ImageToField::create_latvol_vector_field2(ITKDatatypeHandle &img){
  typedef GenericField<LVMesh, HexTrilinearLgn<Vector>, FData3d<Vector, LVMesh> > LVF;
  typedef itk::Image<itk::RGBPixel<Type>,3> InputImageType;
  
  InputImageType *n = dynamic_cast< InputImageType * >( img.get_rep()->data_.GetPointer() );
  
  // get number of data points
  unsigned int size_x = (n->GetRequestedRegion()).GetSize()[0];
  unsigned int size_y = (n->GetRequestedRegion()).GetSize()[1];
  unsigned int size_z = (n->GetRequestedRegion()).GetSize()[2];
  
  // get spacing between data points
  float space_x = n->GetSpacing()[0];
  float space_y = n->GetSpacing()[1];
  float space_z = n->GetSpacing()[2];
  
  // get origin in physical space
  double origin_x = n->GetOrigin()[0];
  double origin_y = n->GetOrigin()[1];
  double origin_z = n->GetOrigin()[2];
  
  // the origin specified by the itk image should remain the same
  // so we must make the min and max points accordingly
  
  double spread_x = (space_x * (size_x-1));
  double spread_y = (space_y * (size_y-1));
  double spread_z = (space_z * (size_z-1));
  
  Point min(origin_x - spread_x, origin_y - spread_y, origin_z - spread_z);
  Point max(origin_x + spread_x, origin_y + spread_y, origin_z + spread_z);
  
  LVMesh* m = scinew LVMesh(size_x, size_y, size_z, min, max);
  
  LVF::mesh_handle_type mh(m);
  
  FieldHandle fh;
  
  if(gui_copy_.get() == 0) {
    remark("Cannot reference rgb data. Copying instead.");
    gui_copy_.set(1);
  }
  
  // copy the data into a SCIRun LatVolField
  fh = scinew LVF(mh); 
  LVMesh::Node::iterator iter, end;
  mh->begin(iter);
  mh->end(end);
  
  // fill data
  typename InputImageType::IndexType pixelIndex;
  typedef typename InputImageType::PixelType PixelType;
  PixelType pixel;
  LVF* fld = (LVF* )fh.get_rep();
  
  for(int z=0; z < (int)size_z; z++) {
    for(int row=0; row < (int)size_y; row++) {
      for(int col=0; col < (int)size_x; col++) {
	if(iter == end) {
	  return fh;
	}
	pixelIndex[0] = col;
	pixelIndex[1] = row;
	pixelIndex[2] = z;
	
	pixel = n->GetPixel(pixelIndex);
	fld->set_value(Vector(pixel[0], pixel[1], pixel[2]), *iter);
	++iter;
      }
    }
  }
  
  return fh;
}
  
  
  
  

template<class InputImageType >
bool ImageToField::run( itk::Object* obj1) 
{
  InputImageType* n = dynamic_cast< InputImageType * >(obj1);
  if( !n ) {
    return false;
  }
  
  int dim = n->GetImageDimension();
  
  switch(dim) {
    
  case 2:
    ofield_handle_ = create_image_field<InputImageType>(inhandle1_);
    break;
    
  case 3:
    ofield_handle_ = create_latvol_field<InputImageType>(inhandle1_);
    break;
  default:
    error("Cannot convert data that is not 2D or 3D to a SCIRun Field.");
    return false;
  }
  ofield_->send(ofield_handle_);
  return true;
}

template< class Type, unsigned int length, unsigned int dimension > 
bool ImageToField::run2( itk::Object* obj1)
{
  if(dynamic_cast<itk::Image< itk::Vector<Type,length> ,dimension>* >(obj1)) {
    switch(dimension) {
      
    case 2:
      ofield_handle_ = create_image_vector_field1<Type, length>(inhandle1_);
      break;
    case 3:
      ofield_handle_ = create_latvol_vector_field1<Type, length>(inhandle1_);
      break;
    default:
      error("Cannot convert data that is not 2D or 3D to a SCIRun Vector field.");
      return false;
    }
    
    ofield_->send(ofield_handle_);
    return true;
  }
  else {
    return false;
  }

}

template< class Type,  unsigned int dimension > 
bool ImageToField::run3( itk::Object* obj1)
{
  if(dynamic_cast<itk::Image< itk::RGBPixel<Type> ,dimension>* >(obj1)) {
    switch(dimension) {
      
    case 2:
      ofield_handle_ = create_image_vector_field2<Type>(inhandle1_);
      break;
    case 3:
      ofield_handle_ = create_latvol_vector_field2<Type>(inhandle1_);
      break;
    default:
      error("Cannot convert data that is not 2D or 3D to a SCIRun Vector field.");
      return false;
    }
    
    ofield_->send(ofield_handle_);
    return true;
  }
  else {
    return false;
  }

}

  
void ImageToField::execute(){
  inport1_ = (ITKDatatypeIPort *)get_iport("InputImage");
  ofield_ = (FieldOPort *)get_oport("OutputImage");
  
  if (!inport1_) {
    error("Unable to initialize iport 'InputImage'.");
    return;
  }
  if (!ofield_) {
    error("Unable to initialize oport 'OutputImage'.");
    return;
  }

  inport1_->get(inhandle1_);
  
  if(!inhandle1_.get_rep())
    return;
  
  // get input
  itk::Object *n = inhandle1_.get_rep()->data_.GetPointer();
  
  // can we operate on it?
  if(0) { }
  else if(run< itk::Image<float, 2> >(n)) { }
  else if(run< itk::Image<float, 3> >(n)) { }
  else if(run< itk::Image<double, 2> >(n)) { }
  else if(run< itk::Image<double, 3> >(n)) { }
  else if(run< itk::Image<int, 2> >(n)) { }
  else if(run< itk::Image<int, 3> >(n)) { }
  else if(run< itk::Image<unsigned char, 2> >(n)) { }
  else if(run< itk::Image<unsigned char, 3> >(n)) { }
  else if(run< itk::Image<char, 2> >(n)) { }
  else if(run< itk::Image<char, 3> >(n)) { }
  else if(run< itk::Image<unsigned short, 2> >(n)) { }
  else if(run< itk::Image<unsigned short, 3> >(n)) { }
  else if(run< itk::Image<short, 2> >(n)) { }
  else if(run< itk::Image<short, 3> >(n)) { }
  else if(run< itk::Image<unsigned long, 2> >(n)) { }
  else if(run< itk::Image<unsigned long, 3> >(n)) { }
  else if(run2< unsigned char, 3, 2 >(n)) { }
  else if(run2< unsigned char, 3, 3 >(n)) { }
  else if(run2< char, 3, 2 >(n)) { }
  else if(run2< char, 3, 3 >(n)) { }
  else if(run2< float, 3, 2 >(n)) { }
  else if(run2< float, 3, 3 >(n)) { }
  else if(run2< double, 3, 2 >(n)) { }
  else if(run2< double, 3, 3 >(n)) { }
  else if(run2< int, 3, 2 >(n)) { }
  else if(run2< int, 3, 3 >(n)) { }
  else if(run2< unsigned short, 3, 2 >(n)) { }
  else if(run2< unsigned short, 3, 3 >(n)) { }
  else if(run2< short, 3, 2 >(n)) { }
  else if(run2< short, 3, 3 >(n)) { }
  else if(run2< unsigned long, 3, 2 >(n)) { }
  else if(run2< unsigned long, 3, 3 >(n)) { }
  else if(run3< unsigned char, 2 >(n)) { }
  else if(run3< unsigned char, 3 >(n)) { }
  else if(run3< char, 2 >(n)) { }
  else if(run3< char, 3 >(n)) { }
  else if(run3< float, 2 >(n)) { }
  else if(run3< float, 3 >(n)) { }
  else if(run3< double, 2 >(n)) { }
  else if(run3< double, 3 >(n)) { }
  else if(run3< int, 2 >(n)) { }
  else if(run3< int, 3 >(n)) { }
  else if(run3< unsigned short, 2 >(n)) { }
  else if(run3< unsigned short, 3 >(n)) { }
  else if(run3< short, 2 >(n)) { }
  else if(run3< short, 3 >(n)) { }
  else if(run3< unsigned long, 2 >(n)) { }
  else if(run3< unsigned long, 3 >(n)) { }
  else {
    // error
    error("Incorrect input type");
    return;
  }
}

void ImageToField::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}



} // End namespace Insight



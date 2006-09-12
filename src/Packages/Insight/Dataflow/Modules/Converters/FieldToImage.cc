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
 *  FieldToImage.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/TypeDescription.h>
#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/ITKImageField.h>
#include <Core/Datatypes/ITKLatVolField.h>

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Containers/FData.h>

#include "itkVector.h"

namespace Insight {

using namespace SCIRun;

enum FieldType {LATVOLFIELD, ITKLATVOLFIELD, IMAGEFIELD, ITKIMAGEFIELD};

class FieldToImage : public Module {

typedef LatVolMesh<HexTrilinearLgn<Point> >         LVMesh;
typedef ImageMesh<QuadBilinearLgn<Point> >          IMesh;

public:
  FieldIPort* infield_;
  FieldHandle infield_handle_;

  ITKDatatypeOPort* outimage_;
  ITKDatatypeHandle outimage_handle_;
  ITKDatatype* img_;
  GuiInt       copy_;


  FieldToImage(GuiContext*);

  virtual ~FieldToImage();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class data > 
  bool run( const FieldHandle &fh );

  bool run_vector( const FieldHandle &fh );

};


DECLARE_MAKER(FieldToImage)
FieldToImage::FieldToImage(GuiContext* ctx)
  : Module("FieldToImage", ctx, Source, "Converters", "Insight"),
    copy_(get_ctx()->subVar("copy"))
{
  img_ = scinew ITKDatatype;
}

FieldToImage::~FieldToImage(){
}


template<class data >
bool FieldToImage::run( const FieldHandle &fh) 
{
  FieldType current_type;

  typedef HexTrilinearLgn<data> LVBasis;
  typedef GenericField<LVMesh, LVBasis, FData3d<data, LVMesh> > LVF;

  typedef ITKLatVolField<data > ITKLVF;

  typedef QuadBilinearLgn<data> IBasis;
  typedef GenericField<IMesh, IBasis, FData2d<data, IMesh> > IF;
  typedef ITKImageField<data> ITKIF;

  string mesh_type = fh->get_type_description(Field::MESH_TD_E)->get_name();

  // try to dynamically cast to the itk types first, then use
  // get_type_description to see if they are regular latvols or images
  if(dynamic_cast<ITKLVF*>(fh.get_rep())) {
    current_type = ITKLATVOLFIELD;
  }
  else if(dynamic_cast<ITKIF*>(fh.get_rep())) {
    current_type = ITKIMAGEFIELD;
  }
  else if( mesh_type.find("LatVolMesh") != string::npos) {
    current_type = LATVOLFIELD;
  }
  if( mesh_type.find("ImageMesh") != string::npos) {
    current_type = IMAGEFIELD;
  }
  else {
    return false;
  }

  if(current_type == LATVOLFIELD) {

    typedef typename itk::Image<typename LVF::value_type, 3> ImageType;
    LVF* f = dynamic_cast< LVF* >(fh.get_rep());

    // create a new itk image
    typename ImageType::Pointer img = ImageType::New(); 

    // set size
    typename ImageType::SizeType fixedSize = {{f->fdata().dim3(), f->fdata().dim2(), f->fdata().dim1()}};
    img->SetRegions( fixedSize );

    // set origin and spacing
    const BBox bbox = f->mesh()->get_bounding_box();
    Point mesh_center;
    Vector mesh_size;
    if(bbox.valid()) {
      mesh_center = bbox.center();
      mesh_size = bbox.diagonal();
    }
    else {
      error("No bounding box to get center");
      return false;
    }
    
    double origin[ ImageType::ImageDimension ];
    Point bbox_min = bbox.min();
    origin[0] = bbox_min.x();
    origin[1] = bbox_min.y();
    origin[2] = bbox_min.z();
    
    img->SetOrigin( origin );
    
    double spacing[ ImageType::ImageDimension ];
    spacing[0] = mesh_size.x()/f->fdata().dim3();
    spacing[1] = mesh_size.y()/f->fdata().dim2();
    spacing[2] = mesh_size.z()/f->fdata().dim1();
    
    img->SetSpacing( spacing );

    // set new data container
    typename LVF::value_type* imageData = &f->fdata()(0,0,0);
    unsigned long size = (unsigned long)f->fdata().size();

    
    img->GetPixelContainer()->SetImportPointer(imageData, size, false);

    // send the data downstream
    img_->data_ = img;
    outimage_handle_ = img_;

  }
  else if(current_type == ITKLATVOLFIELD) {
    // unwrap it
    img_->data_ = dynamic_cast<ITKLVF*>(fh.get_rep())->get_image();
  }
  else if(current_type == IMAGEFIELD) {
    typedef typename itk::Image<typename IF::value_type, 2> ImageType;
    IF* f = dynamic_cast< IF* >(fh.get_rep());

    // create a new itk image
    typename ImageType::Pointer img = ImageType::New(); 

    // set size
    typename ImageType::SizeType fixedSize = {{f->fdata().dim2(), f->fdata().dim1()}};
    img->SetRegions( fixedSize );

    // set origin and spacing
    const BBox bbox = f->mesh()->get_bounding_box();
    Point mesh_center;
    Vector mesh_size;
    if(bbox.valid()) {
      mesh_center = bbox.center();
      mesh_size = bbox.diagonal();
    }
    else {
      error("No bounding box to get center");
      return false;
    }
    
    double origin[ ImageType::ImageDimension ];
    Point bbox_min = bbox.min();
    origin[0] = bbox_min.x();
    origin[1] = bbox_min.y();
    
    img->SetOrigin( origin );
    
    double spacing[ ImageType::ImageDimension ];
    spacing[0] = mesh_size.x()/f->fdata().dim2();
    spacing[1] = mesh_size.y()/f->fdata().dim1();
    
    img->SetSpacing( spacing );

    // set new data container
    typename IF::value_type* imageData = &f->fdata()(0,0);
    unsigned long size = (unsigned long)f->fdata().size();

    
    img->GetPixelContainer()->SetImportPointer(imageData, size, false);

    // send the data downstream
    img_->data_ = img;
    outimage_handle_ = img_;

  }
  else if(current_type == ITKIMAGEFIELD) {
    // unwrap it
    img_->data_ = dynamic_cast<ITKIF*>(fh.get_rep())->get_image();
  }
  return true;
}

bool FieldToImage::run_vector( const FieldHandle &fh) 
{
  typedef HexTrilinearLgn<Vector> LVBasis;
  typedef GenericField<LVMesh, LVBasis, FData3d<Vector, LVMesh> > LVF;

  typedef ITKLatVolField<Vector > ITKLVF;

  typedef QuadBilinearLgn<Vector> IBasis;
  typedef GenericField<IMesh, IBasis, FData2d<Vector, IMesh> > IF;
  typedef ITKImageField<Vector> ITKIF;

  FieldType current_type;

  string mesh_type = fh->get_type_description(Field::MESH_TD_E)->get_name();

  // try to dynamically cast to the itk types first, then use
  // get_type_description to see if they are regular latvols or images

  if(dynamic_cast<ITKLVF*>(fh.get_rep())) {
    current_type = ITKLATVOLFIELD;
  }
  else if(dynamic_cast<ITKIF*>(fh.get_rep())) {
    current_type = ITKIMAGEFIELD;
  }
  else if(mesh_type.find("LatVolMesh") != string::npos) { 
    current_type = LATVOLFIELD;
  }
  else if(mesh_type.find("ImageMesh") != string::npos) { 
    current_type = IMAGEFIELD;
  }

  else {
    return false;
  }

  if(current_type == LATVOLFIELD) {
    typedef itk::Image<itk::Vector<double>, 3> ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    LVF* f = dynamic_cast< LVF* >(fh.get_rep());

    // create a new itk image
    ImageType::Pointer img = ImageType::New(); 

    // set size
    ImageType::SizeType fixedSize = {{f->fdata().dim3(), f->fdata().dim2(), f->fdata().dim1()}};
    img->SetRegions( fixedSize );

    ImageType::RegionType region;
    
    ImageType::IndexType start;
    
    for(int i=0; i<3; i++)
      start[i] = 0;

    region.SetSize( fixedSize );
    region.SetIndex( start );
    
    img->SetRegions( region );
    img->Allocate();
    
    // set origin and spacing
    const BBox bbox = f->mesh()->get_bounding_box();
    Point mesh_center;
    Vector mesh_size;
    if(bbox.valid()) {
      mesh_center = bbox.center();
      mesh_size = bbox.diagonal();
    }
    else {
      error("No bounding box to get center");
      return false;
    }
    
    double origin[ ImageType::ImageDimension ];
    Point bbox_min = bbox.min();
    origin[0] = bbox_min.x();
    origin[1] = bbox_min.y();
    origin[2] = bbox_min.z();
    
    img->SetOrigin( origin );
    
    double spacing[ ImageType::ImageDimension ];
    spacing[0] = mesh_size.x()/f->fdata().dim3();
    spacing[1] = mesh_size.y()/f->fdata().dim2();
    spacing[2] = mesh_size.z()/f->fdata().dim1();
    
    img->SetSpacing( spacing );

    // copy the data
    LVMesh::Node::iterator iter, end;
    LVF::mesh_handle_type mh = f->get_typed_mesh(); 
    //LVF::handle_type mh((LVMesh*)(f->mesh().get_rep()));
    mh->begin(iter);
    mh->end(end);

    typedef ImageType::PixelType PixelType;
    PixelType pixel;
    LVF* fld = (LVF* )fh.get_rep();

    IteratorType img_iter(img, img->GetRequestedRegion());
    img_iter.GoToBegin();

    while(iter != end ) {
      Vector val;
      if (fld->value(val, *iter)) {	  
	itk::Vector<double> new_val;
	new_val[0] = val[0];
	new_val[1] = val[1];
	new_val[2] = val[2];
	
	img_iter.Set(new_val);
      } 
      ++iter;
      img_iter.operator++();
    }

    // send the data downstream
    img_->data_ = img;
    outimage_handle_ = img_;

  }
  else if(current_type == ITKLATVOLFIELD) {
    // unwrap it
    img_->data_ = dynamic_cast<ITKLVF*>(fh.get_rep())->get_image();
  }
  else if(current_type == IMAGEFIELD) {
    typedef itk::Image< itk::Vector<double>, 2> ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    IF* f = dynamic_cast< IF* >(fh.get_rep());

    // create a new itk image
    ImageType::Pointer img = ImageType::New(); 

    // set size
    ImageType::SizeType fixedSize = {{f->fdata().dim2(), f->fdata().dim1()}};
    img->SetRegions( fixedSize );

    ImageType::RegionType region;
    
    ImageType::IndexType start;
    
    for(int i=0; i<3; i++)
      start[i] = 0;

    region.SetSize( fixedSize );
    region.SetIndex( start );
    
    img->SetRegions( region );
    img->Allocate();

    // set origin and spacing
    const BBox bbox = f->mesh()->get_bounding_box();
    Point mesh_center;
    Vector mesh_size;
    if(bbox.valid()) {
      mesh_center = bbox.center();
      mesh_size = bbox.diagonal();
    }
    else {
      error("No bounding box to get center");
      return false;
    }
    
    double origin[ ImageType::ImageDimension ];
    Point bbox_min = bbox.min();
    origin[0] = bbox_min.x();
    origin[1] = bbox_min.y();
    
    img->SetOrigin( origin );
    
    double spacing[ ImageType::ImageDimension ];
    spacing[0] = mesh_size.x()/f->fdata().dim2();
    spacing[1] = mesh_size.y()/f->fdata().dim1();
    
    img->SetSpacing( spacing );

     // copy the data
    IMesh::Node::iterator iter, end;
    IMesh::handle_type mh((IMesh*)(f->mesh().get_rep()));
    mh->begin(iter);
    mh->end(end);

    typedef ImageType::PixelType PixelType;
    PixelType pixel;
    IF* fld = (IF* )fh.get_rep();

    IteratorType img_iter(img, img->GetRequestedRegion());
    img_iter.GoToBegin();

    while(iter != end ) {
      Vector val;
      if (fld->value(val, *iter)) {	  
	itk::Vector<double> new_val;
	new_val[0] = val[0];
	new_val[1] = val[1];
	new_val[2] = val[2];
	
	img_iter.Set(new_val);
      } 
      ++iter;
      img_iter.operator++();
    }   
    // send the data downstream
    img_->data_ = img;
    outimage_handle_ = img_;

  }
  else if(current_type == ITKIMAGEFIELD) {
    // unwrap it
    img_->data_ = dynamic_cast<ITKIF*>(fh.get_rep())->get_image();
  }
  return true;
}

void FieldToImage::execute(){

  infield_ = (FieldIPort *)get_iport("InputField");
  outimage_ = (ITKDatatypeOPort *)get_oport("OutputImage");

  if(!infield_) {
    error("Unable to initialize iport 'InputField'.");
    return;
  }
  if(!outimage_) {
    error("Unable to initialize oport 'OutputImage'.");
    return;
  }

  if(!infield_->get(infield_handle_)) {
    error("No data in InputField port");
    return;
  }

  // Determine which type of field we are convertion from.
  // Our only options are ImageField, ITKImageField,
  // LatVolField and ITKLatVolField
  if(0) { }
  else if(run<double>(infield_handle_)) {}
  else if(run<float>(infield_handle_)) {}
  else if(run<unsigned char>(infield_handle_)) {}
  else if(run<unsigned short>(infield_handle_)) {}
  else if(run_vector(infield_handle_)) {}
  else {
    error("Unknown type");
    return;
  }
  outimage_->send(outimage_handle_);  
}

void FieldToImage::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Insight



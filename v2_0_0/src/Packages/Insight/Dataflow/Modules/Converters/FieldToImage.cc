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
 *  FieldToImage.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geometry/BBox.h>
#include <Packages/Insight/Core/Datatypes/ITKImageField.h>
#include <Packages/Insight/Core/Datatypes/ITKLatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/LatVolField.h>

#include <Core/Datatypes/ImageMesh.h>

namespace Insight {

using namespace SCIRun;

enum FieldType {LATVOLFIELD, ITKLATVOLFIELD, IMAGEFIELD, ITKIMAGEFIELD};

class PSECORESHARE FieldToImage : public Module {
public:
  FieldIPort* infield_;
  FieldHandle infield_handle_;

  ITKDatatypeOPort* outimage_;
  ITKDatatypeHandle outimage_handle_;
  ITKDatatype* img_;


  FieldToImage(GuiContext*);

  virtual ~FieldToImage();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  // Run function will dynamically cast data to determine which
  // instantiation we are working with. The last template type
  // refers to the last template type of the filter intstantiation.
  template< class data > 
  bool run( const FieldHandle &fh );

};


DECLARE_MAKER(FieldToImage)
FieldToImage::FieldToImage(GuiContext* ctx)
  : Module("FieldToImage", ctx, Source, "Converters", "Insight")
{
  img_ = scinew ITKDatatype;
}

FieldToImage::~FieldToImage(){
}


template<class data >
bool FieldToImage::run( const FieldHandle &fh) 
{
  FieldType current_type;

  if(dynamic_cast<LatVolField< data >*>(fh.get_rep())) {
    current_type = LATVOLFIELD;
  }
  else if(dynamic_cast<ITKLatVolField< data >*>(fh.get_rep())) {
    current_type = ITKLATVOLFIELD;
  }
  else if(dynamic_cast<ImageField< data >*>(fh.get_rep())) {
    current_type = IMAGEFIELD;
  }
  else if(dynamic_cast<ITKImageField< data >*>(fh.get_rep())) {
    current_type = ITKIMAGEFIELD;
  }
  else {
    return false;
  }

  if(current_type == LATVOLFIELD) {
    typedef LatVolField< data > LatVolFieldType;
    typedef typename itk::Image<typename LatVolFieldType::value_type, 3> ImageType;
    LatVolFieldType* f = dynamic_cast< LatVolFieldType* >(fh.get_rep());

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
    origin[0] = mesh_center.x();
    origin[1] = mesh_center.y();
    origin[2] = mesh_center.z();
    
    img->SetOrigin( origin );
    
    double spacing[ ImageType::ImageDimension ];
    spacing[0] = mesh_size.x()/f->fdata().dim3();
    spacing[1] = mesh_size.y()/f->fdata().dim2();
    spacing[2] = mesh_size.z()/f->fdata().dim1();
    
    img->SetSpacing( spacing );

    // set new data container
    typename LatVolFieldType::value_type* imageData = &f->fdata()(0,0,0);
    unsigned long size = (unsigned long)f->fdata().size();

    
    img->GetPixelContainer()->SetImportPointer(imageData, size, true);

    // send the data downstream
    img_->data_ = img;
    outimage_handle_ = img_;

  }
  else if(current_type == ITKLATVOLFIELD) {
    // unwrap it
    img_->data_ = dynamic_cast<ITKLatVolField< data >*>(fh.get_rep())->get_image();
  }
  else if(current_type == IMAGEFIELD) {
    typedef ImageField< data > ImageFieldType;
    typedef typename itk::Image<typename ImageFieldType::value_type, 2> ImageType;
    ImageFieldType* f = dynamic_cast< ImageFieldType* >(fh.get_rep());

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
    origin[0] = mesh_center.x();
    origin[1] = mesh_center.y();
    
    img->SetOrigin( origin );
    
    double spacing[ ImageType::ImageDimension ];
    spacing[0] = mesh_size.x()/f->fdata().dim2();
    spacing[1] = mesh_size.y()/f->fdata().dim1();
    
    img->SetSpacing( spacing );

    // set new data container
    typename ImageFieldType::value_type* imageData = &f->fdata()(0,0);
    unsigned long size = (unsigned long)f->fdata().size();

    
    img->GetPixelContainer()->SetImportPointer(imageData, size, true);

    // send the data downstream
    img_->data_ = img;
    outimage_handle_ = img_;

  }
  else if(current_type == ITKIMAGEFIELD) {
    // unwrap it
    img_->data_ = dynamic_cast<ITKImageField< data >*>(fh.get_rep())->get_image();
  }
  else {
    // should never get here
    return false;
  }

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



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
    // allocate a new itk image
    typedef LatVolField< data > LatVolFieldType;
    typedef itk::Image<LatVolFieldType::value_type, 3> ImageType;
    
    LatVolFieldType* fld = (LatVolFieldType*)infield_handle_.get_rep();
    ImageType::Pointer img = ImageType::New();
    
    // image start index
    ImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;
    
    // image size
    unsigned int size_x = fld->fdata().dim1();
    unsigned int size_y = fld->fdata().dim2();
    unsigned int size_z = fld->fdata().dim3();
    
    ImageType::SizeType size;
    size[0] = size_x;
    size[1] = size_y;
    size[2] = size_z;
    
    // allocate image
    ImageType::RegionType region;
    region.SetSize( size );
    region.SetIndex( start );
    
    img->SetRegions( region );
    img->Allocate();
    
    // image origin and spacing
    const BBox bbox = fld->mesh()->get_bounding_box();
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
    spacing[0] = mesh_size.x()/size_x;
    spacing[1] = mesh_size.y()/size_y;
    spacing[2] = mesh_size.z()/size_z;
    
    img->SetSpacing( spacing );
    
    // iterate through the field and copy data
    ImageType::IndexType pixelIndex;
    LatVolFieldType::value_type value;
    
    FData3d<double>::iterator iter, end;
    iter = fld->fdata().begin();
    end = fld->fdata().end();
    
    for(int z = 0; z < size_z; z++) {
      for(int row=0; row < size_y; row++) {
	for(int col=0; col < size_x; col++) {
	  if(iter == end) {
	    error("Reached end before all data was filled");
	    return false;
	  }
	  value = *iter;
	  
	  pixelIndex[0] = col;
	  pixelIndex[1] = row;
	  pixelIndex[2] = z;
	  
	  img->SetPixel(pixelIndex, value);
	  iter++;
	}
      }
    }
    // send itk image downstream
    img_->data_ = img;
    outimage_handle_ = img_;
    
  }
  else if(current_type == ITKLATVOLFIELD) {

  }
  else if(current_type == IMAGEFIELD) {

  }
  else if(current_type == ITKIMAGEFIELD) {

  }
  else {
    // should never get here
    return false;
  }

  /*
  // allocate a new itk image
  typedef LatVolField< double > LatVolFieldType;
  typedef itk::Image<LatVolFieldType::value_type, 3> ImageType;
  
  LatVolFieldType* fld = (LatVolFieldType*)infield_handle_.get_rep();

  std::cout << "*** TYPE " << fld->type_name(0) << std::endl;
  ImageType::Pointer img = ImageType::New();

  // image start index
  ImageType::IndexType start;
  start[0] = 0;
  start[1] = 0;
  start[2] = 0;

  // image size
  unsigned int size_x = fld->fdata().dim1();
  unsigned int size_y = fld->fdata().dim2();
  unsigned int size_z = fld->fdata().dim3();

  ImageType::SizeType size;
  size[0] = size_x;
  size[1] = size_y;
  size[2] = size_z;

  // allocate image
  ImageType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );
  
  img->SetRegions( region );
  img->Allocate();
  
  // image origin and spacing
  const BBox bbox = fld->mesh()->get_bounding_box();
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
  spacing[0] = mesh_size.x()/size_x;
  spacing[1] = mesh_size.y()/size_y;
  spacing[2] = mesh_size.z()/size_z;

  img->SetSpacing( spacing );
  
  // iterate through the field and copy data
  ImageType::IndexType pixelIndex;
  LatVolFieldType::value_type value;
  
  FData3d<double>::iterator iter, end;
  iter = fld->fdata().begin();
  end = fld->fdata().end();
  
  for(int z = 0; z < size_z; z++) {
    for(int row=0; row < size_y; row++) {
      for(int col=0; col < size_x; col++) {
	if(iter == end) {
	  error("Reached end before all data was filled");
	  return false;
	}
	value = *iter;

	pixelIndex[0] = col;
	pixelIndex[1] = row;
	pixelIndex[2] = z;
	
	img->SetPixel(pixelIndex, value);
	iter++;
      }
    }
  }


  // send itk image downstream
  img_->data_ = img;
  outimage_handle_ = img_;

  outimage_->send(outimage_handle_);

  */
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



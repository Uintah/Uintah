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
 * C++ (CC) FILE : DicomImage.cc
 *
 * DESCRIPTION   : A DicomImage object contains all of the data and 
 *                 information relevant to a single DICOM series.  This 
 *                 includes the pixel buffer, dimension, size along each 
 *                 axis, origin, pixel spacing, and index. This object is
 *                 typically initialized using the DicomSeriesReader.
 *                       
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *                 
 * CREATED       : 9/19/2003
 * MODIFIED      : 10/3/2003
 * DOCUMENTATION :
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/
  
// SCIRun includes
#include <Core/Algorithms/DataIO/DicomImage.h>

using namespace std;

namespace SCIRun {

/*===========================================================================*/
// 
// DicomImage
//
// Description : Constructor
//
// Arguments   : none
//
DicomImage::DicomImage()
{
  pixel_buffer_ = 0;
  size_ = 0;
  origin_ = 0;
  spacing_ = 0;
  index_ = 0;
  id_ = "";
}
  
/*===========================================================================*/
// 
// DicomImage
//
// Description : Constructor
//
// Arguments   : none
//
 
DicomImage::DicomImage( itk::DicomImageIO::Pointer io, 
                        ImageNDType::Pointer image, string id )
{
  ImageNDType::RegionType region = image->GetLargestPossibleRegion();

  // Initialize all member variables
  id_ = id;
  num_pixels_ = region.GetNumberOfPixels();
  pixel_buffer_ = scinew PixelType[num_pixels_];
  PixelType * data = image->GetPixelContainer()->GetBufferPointer();

  memcpy( pixel_buffer_, data, num_pixels_ * sizeof(PixelType) );

  const std::type_info& type = io->GetPixelType();

  if( type == typeid(short) )
  {
    data_type_ = "SHORT";
  }
  else if( type == typeid(unsigned short) )
  {
    data_type_ = "USHORT";
  }
  else if( type == typeid(char) )
  {
    data_type_ = "CHAR";
  }
  else if( type == typeid(unsigned char) )
  {
    data_type_ = "UCHAR";
  }
  else
  {
    data_type_ = "UNKNOWN";
  }

  dim_ = region.GetImageDimension();

  // Make sure single files have dimension 2, not 3
  if( dim_ == 3 ) 
  {
    if( region.GetSize(2) == 1)
    {
      dim_ = 2;
    }
  }

  size_ = scinew int[dim_];
  origin_ = scinew double[dim_];
  spacing_ = scinew double[dim_];
  index_ = scinew int[dim_];

  for( int j = 0; j < dim_; j++ )
  {
    size_[j] = region.GetSize(j);
    origin_[j] = image->GetOrigin()[j];
    spacing_[j] = image->GetSpacing()[j];
    index_[j] =  region.GetIndex(j); 
  }

}

/*===========================================================================*/
// 
// DicomImage
//
// Description : Copy constructor
//
// Arguments   : 
//
// const DicomImage& d - DicomImage object to be copied
//
DicomImage::DicomImage(const DicomImage& d)
{
  id_ = d.id_;
  num_pixels_ = d.num_pixels_;
  pixel_buffer_ = scinew PixelType[num_pixels_];
  PixelType * dpb = d.pixel_buffer_;
  for( unsigned long i = 0; i < num_pixels_; i++ )
  {
    pixel_buffer_[i] = dpb[i];
  }

  // ??? data_type;

  dim_ = d.dim_;

  size_ = scinew int[dim_];
  origin_ = scinew double[dim_];
  spacing_ = scinew double[dim_];
  index_ = scinew int[dim_];

  for( int j = 0; j < dim_; j++ )
  {
    size_[j] = d.size_[j];
    origin_[j] = d.origin_[j];
    spacing_[j] = d.spacing_[j];
    index_[j] =  d.index_[j]; 
  }
}
 
/*===========================================================================*/
// 
// ~DicomImage
//
// Description : Destructor
//
// Arguments   : none
//
DicomImage::~DicomImage()
{
  // All memory allocation was done using scinew, so no memory needs to be 
  // manually deallocated.
}

/*===========================================================================*/
// 
// get_id
//
// Description : Returns the string id of this Dicom image
//
// Arguments   : none
//
string DicomImage::get_id()
{
  return id_;
}

/*===========================================================================*/
// 
// get_num_pixels
//
// Description : Returns the number of pixels in the pixel buffer.
//
// Arguments   : none
//
int DicomImage::get_num_pixels()
{
  return num_pixels_;
}

/*===========================================================================*/
// 
// get_pixel_buffer
//
// Description : Returns an array of pixel values of type PixelType.  
//
// Arguments   : none
//
PixelType * DicomImage::get_pixel_buffer()
{
  return pixel_buffer_;
}

/*===========================================================================*/
// 
// get_data_type
// 
// Description : Returns the type of data stored at each pixel in the DICOM
//               files, (i.e. float, unsigned short, etc.)  This is the type 
//               of data in the array returned by get_pixel_buffer.
//
// Arguments   : none
//
string DicomImage::get_data_type()
{
  return data_type_;
}

/*===========================================================================*/
// 
// get_dimension
//
// Description : Returns the dimension of the DICOM image (i.e. 2, 3).
//               If only one DICOM file was in the series, the dimension is 
//               2.  If more than one DICOM file was in the series, the 
//               dimension is 3.   
//
// Arguments   : none
//
int DicomImage::get_dimension()
{
  return dim_;
}

/*===========================================================================*/
// 
// get_size
//
// Description : Returns the number of pixels/values in the DICOM image along
//               the axis referred to by "i".  
//
// Arguments   : 
// 
// int i - Refers to an axis in xyz.  
//         0 = x axis
//         1 = y axis
//         2 = z axis
//
int DicomImage::get_size( int i )
{
  assert( i >= 0 && i < dim_ );
  return size_[i];
}

/*===========================================================================*/
// 
// get_origin
//
// Description : Returns one value in the 2D/3D coordinate of the origin of 
//               the DICOM image.  The value returned is the value along the
//               axis referred to by "i".
//
// Arguments   : 
//
// int i - Refers to an axis in xyz.  
//         0 = x axis
//         1 = y axis
//         2 = z axis
//
double DicomImage::get_origin( int i )
{
  //return image_->GetOrigin()[i]; 
  assert( i >= 0 && i < dim_ );
  return origin_[i];
}

/*===========================================================================*/
// 
// get_spacing
//
// Description : Returns the pixel spacing for each dimension of the image.
//
// Arguments   : 
//
// int i - Refers to an axis in xyz.  
//         0 = x axis
//         1 = y axis
//         2 = z axis
//
double DicomImage::get_spacing( int i )
{
  assert( i >= 0 && i < dim_ );
  return spacing_[i];
}

/*===========================================================================*/
// 
// get_index
//
// Description : Returns the index for each dimension of the image.
//
// Arguments   : 
//
// int i - Refers to an axis in xyz.  
//         0 = x axis
//         1 = y axis
//         2 = z axis
//
int DicomImage::get_index( int i )
{
  assert( i >= 0 && i < dim_ );
  return index_[i];
}

/*===========================================================================*/
// 
// print_image_info
//
// Description : Prints image info for this Dicom image.  This is
//               useful for debugging.
//
// Arguments   : none
//
void DicomImage::print_image_info()
{

  // Get data from DICOM files

  // Get id
  string id = get_id();
  cout << "(DicomImage::print_image_info) ID: " << id << "\n";

  // Get number of pixels
  int num_pixels = get_num_pixels();
  cout << "(DicomImage::print_image_info) Num Pixels: " << num_pixels << "\n";

  // Get pixel buffer data (array)
  PixelType min = INT_MAX;
  PixelType max = 0;
  
  PixelType * pixel_data = get_pixel_buffer();
  for( int i = 0; i < num_pixels; i++ )
  {
    //cout << "(DicomImage) Pixel value " << i << ": " << pixel_data[i] 
    //    << "\n"; 
    if( pixel_data[i] < min ) 
    {
      min = pixel_data[i];
      //cout << "(DicomImage) Pixel value " << i << ": " << pixel_data[i] 
      //     << " is new minimum\n"; 
    }
    if( pixel_data[i] > max ) max = pixel_data[i];    
  }

  cout << "(DicomImage::print_image_info) Min, Max: [ " << min 
       << " " << max << " ]\n";

  // Get pixel type
  string data_type = get_data_type();
  cout << "(DicomImage::print_image_info) Data Type: " << data_type << "\n";

  // Get image dimension
  int image_dim = get_dimension();
  cout << "(DicomImage::print_image_info) Dimension: " << image_dim << "\n";

  // Get the size of each axis
  cout << "(DicomImage::print_image_info) Size: [ ";
  for( int j = 0; j < image_dim; j++ )
  {
    cout << get_size(j) << " "; 
  }
  cout << "]\n";

  // Get the origin  
  cout << "(DicomImage::print_image_info) Origin: [ ";
  for( int k = 0; k < image_dim; k++ )
  {
    cout << get_origin(k) << " "; 
  }
  cout << "]\n";

  // Get the pixel spacing
  cout << "(DicomImage::print_image_info) Spacing: [ ";
  for( int m = 0; m < image_dim; m++ )
  {
    cout << get_spacing(m) << " "; 
  }
  cout << "]\n";

  // Get the indices
  cout << "(DicomImage::print_image_info) Index: [ ";
  for( int n = 0; n < image_dim; n++ )
  {
    cout << get_index(n) << " "; 
  }
  cout << "]\n";


}

} // End namespace SCIRun

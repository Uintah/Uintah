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
 * DESCRIPTION   : 
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *                 
 *                 Darby J. Van Uitert
 *                 SCI Institute
 *                 University of Utah
 *
 * CREATED       : 9/19/2003
 * MODIFIED      : 9/19/2003
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
  pixel_buffer = 0;
  size = 0;
  origin = 0;
  spacing = 0;
  index = 0;
  
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
                        ImageNDType::Pointer image )
{
  ImageNDType::RegionType region = image->GetLargestPossibleRegion();

  // Initialize all member variables
  num_pixels = region.GetNumberOfPixels();
  pixel_buffer = scinew PixelType[num_pixels];
  PixelType * data = image->GetPixelContainer()->GetBufferPointer();
  for(unsigned int i=0; i < num_pixels; i++ )
  {
    pixel_buffer[i] = *data++;
  }

  // ??? data_type;

  dim = region.GetImageDimension();

  size = scinew int[dim];
  origin = scinew double[dim];
  spacing = scinew double[dim];
  index = scinew int[dim];

  for( int j = 0; j < dim; j++ )
  {
    size[j] = region.GetSize(j);
    origin[j] = image->GetOrigin()[j];
    spacing[j] = image->GetSpacing()[j];
    index[j] =  region.GetIndex(j); 
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
  num_pixels = d.num_pixels;
  pixel_buffer = scinew PixelType[num_pixels];
  PixelType * dpb = d.pixel_buffer;
  for( unsigned long i = 0; i < num_pixels; i++ )
  {
    pixel_buffer[i] = dpb[i];
  }

  // ??? data_type;

  dim = d.dim;

  size = scinew int[dim];
  origin = scinew double[dim];
  spacing = scinew double[dim];
  index = scinew int[dim];

  for( int j = 0; j < dim; j++ )
  {
    size[j] = d.size[j];
    origin[j] = d.origin[j];
    spacing[j] = d.spacing[j];
    index[j] =  d.index[j]; 
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
// get_num_pixels
//
// Description : Returns the number of pixels in the pixel buffer.
//
// Arguments   : none
//
int DicomImage::get_num_pixels()
{
  return num_pixels;
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
  return pixel_buffer;
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
void DicomImage::get_data_type()
{
  // TODO: Fix this
  //std::type_info type = io_->GetPixelType();
  //io_->GetPixelType();
  //cerr << "Pixel Type: " << io_->GetPixelType();
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
  return dim;
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
  assert( i >= 0 && i < dim );
  return size[i];
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
  assert( i >= 0 && i < dim );
  return origin[i];
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
  assert( i >= 0 && i < dim );
  return spacing[i];
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
  assert( i >= 0 && i < dim );
  return index[i];
}

} // End namespace SCIRun

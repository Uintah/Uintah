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
 * C++ (CC) FILE : AnalyzeImage.cc
 *
 * DESCRIPTION   : A AnalyzeImage object contains all of the data and 
 *                 information relevant to a single set of Analyze files.  
 *                 This includes the pixel buffer, dimension, size along 
 *                 each axis, origin, pixel spacing, and index. This object 
 *                 is typically initialized using the AnalyzeReader.
 *                      
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *                 
 * CREATED       : 9/19/2003
 * MODIFIED      : 10/4/2003
 * DOCUMENTATION :
 * NOTES         : 
 *
 * Copyright (C) 2003 SCI Group
*/
 
// SCIRun includes
#include <Core/Algorithms/DataIO/AnalyzeImage.h>

using namespace std;

namespace SCIRun {

/*===========================================================================*/
// 
// AnalyzeImage
//
// Description : Constructor
//
// Arguments   : none
//
AnalyzeImage::AnalyzeImage()
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
// AnalyzeImage
//
// Description : Constructor
//
// Arguments   : none
//

AnalyzeImage::AnalyzeImage( itk::AnalyzeImageIO::Pointer io, 
                        ImageNDType::Pointer image, string id )
{
  ImageNDType::RegionType region = image->GetLargestPossibleRegion();

  // Initialize all member variables
  id_ = id;
  num_pixels_ = region.GetNumberOfPixels();
  pixel_buffer_ = scinew PixelType[num_pixels_];
  PixelType * data = image->GetPixelContainer()->GetBufferPointer();
  for(unsigned int i=0; i < num_pixels_; i++ )
  {
    pixel_buffer_[i] = *data++;
  }

#if ((ITK_VERSION_MAJOR == 1) && (ITK_VERSION_MINOR >= 8)) || (ITK_VERSION_MAJOR > 1)
  const std::type_info& type = io->GetComponentTypeInfo();
#else
  const std::type_info& type = io->GetPixelType();
#endif


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
  else if( type == typeid(float) )
  {
    data_type_ = "FLOAT";
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
// AnalyzeImage
//
// Description : Copy constructor
//
// Arguments   : 
//
// const AnalyzeImage& d - AnalyzeImage object to be copied
//
AnalyzeImage::AnalyzeImage(const AnalyzeImage& d)
{
  num_pixels_ = d.num_pixels_;
  pixel_buffer_ = scinew PixelType[num_pixels_];
  PixelType * dpb = d.pixel_buffer_;
  for( unsigned long i = 0; i < num_pixels_; i++ )
  {
    pixel_buffer_[i] = dpb[i];
  }

  data_type_ = d.data_type_;

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
// ~AnalyzeImage
//
// Description : Destructor
//
// Arguments   : none
//
AnalyzeImage::~AnalyzeImage()
{
  // Deallocate all dynamically allocated memory
  /*
  if( pixel_buffer_ != 0 ) delete [] pixel_buffer_;
  if( size_ != 0 ) delete [] size_;
  if( origin_ != 0 ) delete [] origin_;
  if( spacing_ != 0 ) delete [] spacing_;
  if( index_ != 0 ) delete [] index_;
  */
}

/*===========================================================================*/
// 
// get_id
//
// Description : Returns the string id of this Dicom image
//
// Arguments   : none
//
string AnalyzeImage::get_id()
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
int AnalyzeImage::get_num_pixels()
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
PixelType * AnalyzeImage::get_pixel_buffer()
{
  return pixel_buffer_;
}

/*===========================================================================*/
// 
// get_data_type
//
// Description : Returns the type of data stored at each pixel in the ANALYZE
//               files, (i.e. float, unsigned short, etc.)  This is the type 
//               of data in the array returned by get_pixel_buffer.
//
// Arguments   : none
//
std::string AnalyzeImage::get_data_type()
{
  return data_type_;
}

/*===========================================================================*/
// 
// get_dimension
//
// Description : Returns the dimension of the ANALYZE image (i.e. 2, 3).
//               If only one ANALYZE file was in the series, the dimension is 
//               2.  If more than one ANALYZE file was in the series, the 
//               dimension is 3.   
//
// Arguments   : none
//
int AnalyzeImage::get_dimension()
{
  return dim_;
}

/*===========================================================================*/
// 
// get_size
//
// Description : Returns the number of pixels/values in the ANALYZE image along
//               the axis referred to by "i".  
//
// Arguments   : 
// 
// int i - Refers to an axis in xyz.  
//         0 = x axis
//         1 = y axis
//         2 = z axis
//
int AnalyzeImage::get_size( int i )
{
  assert( i >= 0 && i < dim_ );
  return size_[i];
}

/*===========================================================================*/
// 
// get_origin
//
// Description : Returns one value in the 2D/3D coordinate of the origin of 
//               the ANALYZE image.  The value returned is the value along the
//               axis referred to by "i".
//
// Arguments   : 
//
// int i - Refers to an axis in xyz.  
//         0 = x axis
//         1 = y axis
//         2 = z axis
//
double AnalyzeImage::get_origin( int i )
{
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
double AnalyzeImage::get_spacing( int i )
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
int AnalyzeImage::get_index( int i )
{
  assert( i >= 0 && i < dim_ );
  return index_[i];
}

/*===========================================================================*/
// 
// print_image_info
//
// Description : Prints image info for this Analyze image.  This is
//               useful for debugging.
//
// Arguments   : none
//
void AnalyzeImage::print_image_info()
{

  // Get data from ANALYZE files

  // Get id
  string id = get_id();
  cout << "(AnalyzeImage::print_image_info) ID: " << id << endl;

  // Get number of pixels
  int num_pixels = get_num_pixels();
  cout << "(AnalyzeImage::print_image_info) Num Pixels: " << num_pixels << endl;

  // Get pixel buffer data (array)
  //PixelType * pixel_data = get_pixel_buffer();
  //for( int i = 0; i < num_pixels; i++ )
  // {
  //  cout << "(AnalyzeImage) Pixel value " << i << ": " << pixel_data[i] 
  //      << endl; 
  //}

  // Get pixel type
  //get_data_type();

  // Get image dimension
  int image_dim = get_dimension();
  cout << "(AnalyzeImage::print_image_info) Dimension: " << image_dim << endl;

  // Get the size of each axis
  cout << "(AnalyzeImage::print_image_info) Size: [ ";
  for( int j = 0; j < image_dim; j++ )
  {
    cout << get_size(j) << " "; 
  }
  cout << "]" << endl;

  // Get the origin  
  cout << "(AnalyzeImage::print_image_info) Origin: [ ";
  for( int k = 0; k < image_dim; k++ )
  {
    cout << get_origin(k) << " "; 
  }
  cout << "]" << endl;

  // Get the pixel spacing
  cout << "(AnalyzeImage::print_image_info) Spacing: [ ";
  for( int m = 0; m < image_dim; m++ )
  {
    cout << get_spacing(m) << " "; 
  }
  cout << "]" << endl;

  // Get the indices
  cout << "(AnalyzeImage::print_image_info) Index: [ ";
  for( int n = 0; n < image_dim; n++ )
  { 
    cout << get_index(n) << " "; 
  }
  cout << "]" << endl;
 

}

} // End namespace SCIRun

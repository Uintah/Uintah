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
 * HEADER (H) FILE : AnalyzeImage.h
 *
 * DESCRIPTION     : A AnalyzeImage object contains all of the data and 
 *                   information relevant to a single set of Analyze files.  
 *                   This includes the pixel buffer, dimension, size along 
 *                   each axis, origin, pixel spacing, and index. This object 
 *                   is typically initialized using the AnalyzeReader.
 *                     
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 *                 
 * CREATED         : 9/19/2003
 * MODIFIED        : 10/4/2003
 * DOCUMENTATION   :
 * NOTES           : 
 *
 * Copyright (C) 2003 SCI Group
*/

#ifndef AnalyzeImage_h
#define AnalyzeImage_h

// SCIRun includes
#include <Core/Malloc/Allocator.h>

// Itk includes
#include "itkAnalyzeImageIOFactory.h"
#include "itkAnalyzeImageIO.h"
#include "itkImageSeriesReader.h"

// Standard lib includes
#include <iostream>
#include <assert.h>

namespace SCIRun {

// ****************************************************************************
// **************************** Class: AnalyzeImage ***************************
// ****************************************************************************

typedef float PixelType;
typedef itk::Image<PixelType, 3> ImageNDType;

class AnalyzeImage
{

public:
  // !Constructors
  AnalyzeImage();
  AnalyzeImage( itk::AnalyzeImageIO::Pointer io, ImageNDType::Pointer image,
                std::string id );

  // !Copy constructor
  AnalyzeImage(const AnalyzeImage& d);

  // !Destructor
  ~AnalyzeImage();

  //! Utility functions
  std::string get_id();
  int get_num_pixels();
  PixelType * get_pixel_buffer();
  std::string get_data_type();
  int get_dimension();
  int get_size( int i );
  double get_origin( int i );
  double get_spacing( int i );
  int get_index( int i );
  void print_image_info();

private:
  unsigned long num_pixels_;
  PixelType * pixel_buffer_;
  std::string data_type_;
  int dim_;
  int * size_;
  double * origin_;
  double * spacing_;
  int * index_;
  std::string id_;

protected:

};

} // End namespace SCIRun
 
#endif // AnalyzeImage_h




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
 * HEADER (H) FILE : AnalyzeImage.h
 *
 * DESCRIPTION     : 
 *                     
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 *                 
 *                   Darby J. Van Uitert
 *                   SCI Institute
 *                   University of Utah
 *
 * CREATED         : 9/19/2003
 * MODIFIED        : 9/19/2003
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
#include "itkImageSeriesReader.h"

// Standard lib includes
#include <iostream>
#include <assert.h>

namespace SCIRun {

// ****************************************************************************
// *************************** Class: AnalyzeImage ****************************
// ****************************************************************************

typedef unsigned short PixelType;
typedef itk::Image<PixelType, 3> ImageNDType;

class AnalyzeImage
{

public:
  // !Constructors
  AnalyzeImage();
  //AnalyzeImage( itk::AnalyzeImageIO::Pointer io, ImageNDType::Pointer image );

  // !Copy constructor
  AnalyzeImage(const AnalyzeImage& d);

  // !Destructor
  ~AnalyzeImage();

  // TODO: Implement copy constructor

  //! Utility functions
  int get_num_pixels();
  PixelType * get_pixel_buffer();
  void get_data_type();
  int get_dimension();
  int get_size( int i );
  double get_origin( int i );
  double get_spacing( int i );
  int get_index( int i );

private:
  unsigned long num_pixels;
  PixelType * pixel_buffer;
  // ??? data_type;
  int dim;
  int * size;
  double * origin;
  double * spacing;
  int * index;

protected:

};

} // End namespace SCIRun
 
#endif // AnalyzeImage_h




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
 * HEADER (H) FILE : DicomImage.h
 *
 * DESCRIPTION     : A DicomImage object contains all of the data and 
 *                   information relevant to a single DICOM series.  This 
 *                   includes the pixel buffer, dimension, size along each 
 *                   axis, origin, pixel spacing, and index. This object is
 *                   typically initialized using the DicomSeriesReader.
 *                     
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 *                 
 * CREATED         : 9/19/2003
 * MODIFIED        : 10/3/2003
 * DOCUMENTATION   :
 * NOTES           : 
 *
 * Copyright (C) 2003 SCI Group
*/

#ifndef DicomImage_h
#define DicomImage_h

// SCIRun includes
#include <Core/Malloc/Allocator.h>

// Itk includes
#include "itkDicomImageIOFactory.h"
#include "itkDicomImageIO.h"
#include "itkImageSeriesReader.h"
#include "itkDICOMSeriesFileNames.h"

// Standard lib includes
#include <iostream>
#include <assert.h>

namespace SCIRun {

// ****************************************************************************
// **************************** Class: DicomImage *****************************
// ****************************************************************************

typedef unsigned short PixelType;
typedef itk::Image<PixelType, 3> ImageNDType;

class DicomImage
{

public:
  // !Constructors
  DicomImage();
  DicomImage( itk::DicomImageIO::Pointer io, ImageNDType::Pointer image,
              std::string id );

  // !Copy constructor
  DicomImage(const DicomImage& d);

  // !Destructor
  ~DicomImage();

  //! Utility functions
  std::string get_id();
  int get_num_pixels();
  PixelType * get_pixel_buffer();
  //void get_data_type();
  int get_dimension();
  int get_size( int i );
  double get_origin( int i );
  double get_spacing( int i );
  int get_index( int i );
  void print_image_info();

private:
  unsigned long num_pixels_;
  PixelType * pixel_buffer_;
  // ??? data_type;
  int dim_;
  int * size_;
  double * origin_;
  double * spacing_;
  int * index_;
  std::string id_;

protected:

};

} // End namespace SCIRun
 
#endif // DicomImage_h




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
 * HEADER (H) FILE : DicomSeriesReader.h
 *
 * DESCRIPTION     : Provides ability to read and get information about
 *                   a set of DICOM files.  The read function can read
 *                   one valid DICOM series at a time.  The other functions
 *                   return information about what DICOM series' are in a 
 *                   directory and what files are in each series.  The user
 *                   can use that information to decide which directory/files
 *                   to read from.  The read function stores each DICOM series
 *                   as a DicomImage object, which in turn contains information
 *                   about the series (dimensions, pixel spacing, etc.).
 *                     
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 *                 
 * CREATED         : 9/19/2003
 * MODIFIED        : 10/2/2003
 * DOCUMENTATION   :
 * NOTES           : 
 *
 * Copyright (C) 2003 SCI Group
*/

#ifndef DicomSeriesReader_h
#define DicomSeriesReader_h

// SCIRun includes

// Itk includes
#include "itkDICOMImageIO2.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileReader.h"
#include "itkDICOMSeriesFileNames.h"
#include <Core/Algorithms/DataIO/DicomImage.h>

// Standard lib includes
#include <iostream>

namespace SCIRun {

// ****************************************************************************
// ************************ Class: DicomSeriesReader **************************
// ****************************************************************************

class DicomSeriesReader
{

public:
  DicomSeriesReader();
  ~DicomSeriesReader();

  //! Reading functions
  void set_dir( std::string dir );
  std::string get_dir();
  void set_files( const std::vector<std::string> files );
  const std::vector<std::string> get_files();

  int read( DicomImage & di );

  //! Series information
  const std::vector<std::string> &get_series_uids();
  const std::vector<std::string> &get_file_names( const std::string& 
                                                  series_uid );
  void set_sort_image_num();
  void set_sort_slice_loc();
  void set_sort_pos_patient();

private:

  std::string dir_;
  std::vector<std::string> files_;
  itk::DICOMSeriesFileNames::Pointer names_;

};

} // End namespace SCIRun
 
#endif // DicomSeriesReader_h




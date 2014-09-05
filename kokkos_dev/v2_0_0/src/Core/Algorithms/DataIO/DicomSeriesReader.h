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
#include "itkDicomImageIOFactory.h"
#include "itkDicomImageIO.h"
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




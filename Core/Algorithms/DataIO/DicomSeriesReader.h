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

#ifndef DicomSeriesReader_h
#define DicomSeriesReader_h

// Itk includes
#include "itkDicomImageIOFactory.h"
#include "itkDicomImageIO.h"
#include "itkImageSeriesReader.h"
#include "itkDICOMSeriesFileNames.h"
#include "Testing/Code/BasicFilters/itkFilterWatcher.h"
#include <Core/Algorithms/DataIO/DicomImage.h>

// Standard lib includes
#include <iostream>

namespace SCIRun {

// ****************************************************************************
// ************************ Class: DicomSeriesReader **************************
// ****************************************************************************

typedef itk::ImageSeriesReader<ImageNDType> ReaderType;

class DicomSeriesReader
{

public:
  DicomSeriesReader();
  ~DicomSeriesReader();

  //! Reading function
  DicomImage read_series( char * dir );

private:

protected:

};

} // End namespace SCIRun
 
#endif // DicomSeriesReader_h




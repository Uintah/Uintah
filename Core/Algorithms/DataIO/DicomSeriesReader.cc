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
 * C++ (CC) FILE : DicomSeriesReader.cc
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
#include <Core/Algorithms/DataIO/DicomSeriesReader.h>

using namespace std;

namespace SCIRun {

/*===========================================================================*/
// 
// DicomSeriesReader
//
// Description : Constructor
//
// Arguments   : none
//
DicomSeriesReader::DicomSeriesReader()
{
}

/*===========================================================================*/
// 
// ~DicomSeriesReader
//
// Description : Destructor
//
// Arguments   : none
//
DicomSeriesReader::~DicomSeriesReader()
{
}

/*===========================================================================*/
// 
// read_series
//
// Description : Read in a series of DICOM files and construct a DicomImage
//               object.  Returns the DicomImage object.
//
// Arguments   : none
//
DicomImage DicomSeriesReader::read_series( char * dir )
{
  itk::DicomImageIO::Pointer io = itk::DicomImageIO::New();

  // Get the DICOM filenames from the directory
  itk::DICOMSeriesFileNames::Pointer names = itk::DICOMSeriesFileNames::New();

  // Hard code directory for now
  names->SetDirectory(dir);

  // Create a new reader
  ReaderType::Pointer reader = ReaderType::New();

  // Set reader file names
  reader->SetFileNames(names->GetFileNames());
  reader->SetImageIO(io);
  std::cout << names;

  //FilterWatcher watcher(reader);

  // Check for the ordering specified
  // Hard-code the ordering for now
  int reverse = 1;
  try
  {
    if (reverse)
      {
      reader->ReverseOrderOn();
      }
    reader->Update();
    reader->GetOutput()->Print(std::cout);
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cout << ex;
  }

  ImageNDType::Pointer image = reader->GetOutput();

  return DicomImage( io, image );
}

} // End namespace SCIRun

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
 * DESCRIPTION   : Provides ability to read and get information about
 *                 a set of DICOM files.  The read function can read
 *                 one valid DICOM series at a time.  The other functions
 *                 return information about what DICOM series' are in a 
 *                 directory and what files are in each series.  The user
 *                 can use that information to decide which directory/files
 *                 to read from.  The read function stores each DICOM series
 *                 as a DicomImage object, which in turn contains information
 *                 about the series (dimensions, pixel spacing, etc.).
 *                     
 * AUTHOR(S)     : Jenny Simpson
 *                 SCI Institute
 *                 University of Utah
 *
 * CREATED       : 9/19/2003
 * MODIFIED      : 10/2/2003
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
  dir_ = "";
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
// set_dir 
//
// Description : Set the directory to read from and/or get series information 
//               about.  The directory must be set before any other operations
//               are performed.
//
// Arguments   : 
//
// std::string dir - Directory to read from and/or get series information 
//                   about.
//
void DicomSeriesReader::set_dir( std::string dir )
{
  dir_ = dir;

  // Get the DICOM filenames from the directory
  names_ = itk::DICOMSeriesFileNames::New();

  // Set the directory to read DICOM files from for this series
  names_->SetDirectory( dir_ );
}

/*===========================================================================*/
// 
// get_dir
//
// Description : Returns the directory that was set.
//
// Arguments   : none
//
std::string DicomSeriesReader::get_dir()
{
  return dir_;
}

/*===========================================================================*/
// 
// set_files
//
// Description : Set the files of interest within the directory that should 
//               been set using set_dir().  If no files are set using this
//               function, then all files in the directory will be selected
//               be default.  
//
// Arguments   : 
//
// const std::vector<std::string> files - Selected file names within the 
// directory that was set using set_dir().  These should be file names with
// NO PATH included.  Ex. "IM000123"
//
void DicomSeriesReader::set_files( const std::vector<std::string> files )
{
  if( dir_ == "" )
  {
    cerr << "(DicomSeriesReader::set_files) ERROR: No directory selected.\n";
    return; 
  }

  files_ = files;
  int num_files = files_.size();
  for( int i = 0; i < num_files; i++ )
  {
    files_[i] = string( dir_ + "/" + files_[i] );   
  }
}

/*===========================================================================*/
// 
// get_files
//
// Description : Get the files that have been selected for this series.  If no
//               files were explicitly set using set_files(), then all the 
//               files in the selected directory are returned. 
//
// Arguments   : none
//
const std::vector<std::string> DicomSeriesReader::get_files()
{
  if( dir_ == "" )
  {
    cerr << "(DicomSeriesReader::get_files) WARNING: No directory selected.\n";
  }

  // If specific files were set, return them.  Otherwise, return all the files
  // in the directory.
  if( files_.size() != 0 )
  {
    return files_;
  }
  else
  {
    return names_->GetFileNames();
  } 
}

/*===========================================================================*/
// 
// read
//
// Description : Read in a series of DICOM files and construct a DicomImage
//               object.  A series contains one or more DICOM files. Returns 0 
//               on success, -1 on failure.
//
// Arguments   : 
//
// DicomImage & di - DicomImage object that is initialized in this function and
//                   returned by reference.  This is set to contain all the 
//                   information relevant to this series (i.e. dimensions, 
//                   pixel spacing, etc.).
//
int DicomSeriesReader::read( DicomImage & di )
{
  if( dir_ == "" ) 
  {
    cerr << "(DicomSeriesReader::read) Error: No directory selected.\n";
    return -1;
  } 

  // Get the file names to be read
  std::vector<std::string> read_files;

  // If no files were specified, read all files in the directory 
  if( files_.size() == 0 ) 
  {
    read_files = names_->GetFileNames();
  }
  else
  {
    read_files = files_;
  }

  // Check to see if we have a series or a single file
  int num_files =  read_files.size();
  if( num_files == 1 ) 
  {
    //cerr << "(DicomSeriesReader::read) Reading single file\n";
    itk::DicomImageIO::Pointer io = itk::DicomImageIO::New();

    // Create a new reader
    typedef itk::ImageFileReader<ImageNDType> ReaderType;
    ReaderType::Pointer reader = ReaderType::New();

    // Set reader file names
    reader->SetFileName( const_cast<char*>(read_files[0].c_str()) );

    reader->SetImageIO( io );

    try
    {
      reader->Update();
      //reader->GetOutput()->Print(std::cout);
    }
    catch (itk::ExceptionObject &ex)
    {
      std::cerr << ex;
      return -1;
    }

    ImageNDType::Pointer image = reader->GetOutput();

    di = DicomImage( io, image, read_files[0] );
  }
  else if( num_files > 1 ) 
  { 
    itk::DicomImageIO::Pointer io = itk::DicomImageIO::New();

    // Create a new reader
    typedef itk::ImageSeriesReader<ImageNDType> ReaderType;
    ReaderType::Pointer reader = ReaderType::New();

    // Set reader file names
    reader->SetFileNames( read_files );

    reader->SetImageIO( io );

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
      //reader->GetOutput()->Print(std::cout);
    }
    catch (itk::ExceptionObject &ex)
    {
      std::cerr << ex;
      return -1;
    }

    ImageNDType::Pointer image = reader->GetOutput();

    string id = string( read_files[0] + " ... " + 
                        read_files[read_files.size()-1] );
 
    di = DicomImage( io, image, id );
  }
  else
  {
    cerr << "(DicomSeriesReader::read) ERROR: No files selected.\n";
    return -1;
  }

  return 0;
}

/*===========================================================================*/
// 
// get_series_uids
//
// Description :  Returns a vector containing the UIDs for each series in the 
//                directory that should have been set using set_dir().
//
// Arguments   : 
//
// std::string dir - Full path for directory containing DICOM files
//
const std::vector<std::string> & 
DicomSeriesReader::get_series_uids()
{
  if( dir_ == "" )
  {
    cerr << "(DicomSeriesReader::get_series_uids) Warning: No directory selected.\n";
  }

  return names_->GetSeriesUIDs(); 
}

/*===========================================================================*/
// 
// get_file_names
//
// Description :  Returns a vector containing the file names for a specified
//                series UIDin the directory that should have been set using 
//                set_dir(). The file names are sorted based on the current
//                sorting mode, which is SortByImagePositionPatient. 
//
// Arguments   : none
//
const std::vector<std::string> & 
DicomSeriesReader::get_file_names( const std::string& series_uid )
{
  if( dir_ == "" )
  {
    cerr << "(DicomSeriesReader::get_file_names) Warning: No directory selected.\n";
  }

  return names_->GetFileNames( series_uid );
}

/*===========================================================================*/
// 
// set_sort_image_num
//
// Description :  Set the filename sorting order to sorting images based on the
//                DICOM field of slice number.
//
// Arguments   : none
//
void DicomSeriesReader::set_sort_image_num()
{
  if( dir_ == "" )
  {
    cerr << "(DicomSeriesReader::set_sort_image_num) Warning: No directory selected.\n";
  }

  names_->SetFileNameSortingOrderToSortByImageNumber();
}

/*===========================================================================*/
// 
// set_sort_slice_loc
//
// Description :  Set the filename sorting order to sorting images based on the
//                the DICOM field of slice location.
//
// Arguments   : none
//
void DicomSeriesReader::set_sort_slice_loc()
{
  if( dir_ == "" )
  {
    cerr << "(DicomSeriesReader::set_sort_slice_loc) Warning: No directory selected.\n";
  }

  names_->SetFileNameSortingOrderToSortBySliceLocation();
}

/*===========================================================================*/
// 
// set_sort_pos_patient
//
// Description :  Set the filename sorting order to sorting images based on the
//                the position of the image computed using the 
//                ImagePositionPatient and ImageOrientationPatient DICOM 
//                fields. 
//
// Arguments   : none
//
void DicomSeriesReader::set_sort_pos_patient()
{
  if( dir_ == "" )
  {
    cerr << "(DicomSeriesReader::set_sort_pos_patient) Warning: No directory selected.\n";
  }
 
  names_->SetFileNameSortingOrderToSortByImagePositionPatient();
}

} // End namespace SCIRun

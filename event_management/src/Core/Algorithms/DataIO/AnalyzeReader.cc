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
 * C++ (CC) FILE : AnalyzeReader.cc
 *
 * DESCRIPTION   : Provides ability to read and get information about
 *                 a set of Analyze files.  The read function can read
 *                 one valid Analyze series at a time.  The other functions
 *                 return information about what Analyze series' are in a 
 *                 directory and what files are in each series.  The user
 *                 can use that information to decide which directory/files
 *                 to read from.  The read function stores each Analyze series
 *                 as a AnalyzeImage object, which in turn contains information
 *                 about the series (dimensions, pixel spacing, etc.).
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
#include <Core/Algorithms/DataIO/AnalyzeReader.h>

using namespace std;

namespace SCIRun {

/*===========================================================================*/
// 
// AnalyzeReader
//
// Description : Constructor
//
// Arguments   : none
//
AnalyzeReader::AnalyzeReader()
{
  file_ = "";
}

/*===========================================================================*/
// 
// ~AnalyzeReader
//
// Description : Destructor
//
// Arguments   : none
//
AnalyzeReader::~AnalyzeReader()
{
}

/*===========================================================================*/
// 
// set_file 
//
// Description : Set the Analyze .hdr file to read from.
//
// Arguments   : 
//
// std::string file - Analyze .hdr file to read from.  The full path to the 
//                    file should be included.  The reader assumes that the
//                    image file with the same prefix exists in the same 
//                    directory.  If this is not the case, there will be an
//                    error.
//
//                    Ex: /home/sci/simpson/sci_io/analyze/Brain.hdr
//
void AnalyzeReader::set_file( std::string file )
{
  file_ = file;
}

/*===========================================================================*/
// 
// get_file
//
// Description : Returns the file that was set.
//
// Arguments   : none
//
std::string AnalyzeReader::get_file()
{
  return file_;
}

/*===========================================================================*/
// 
// read
//
// Description : Read in a set (1 .hdr + 1 .img) of Analyze files and 
//               construct a AnalyzeImage object.  Returns 0 on success, -1 on 
//               failure.
//
// Arguments   : 
//
// AnalyzeImage & di - AnalyzeImage object that is initialized in this function
//                     and returned by reference.  This is set to contain all 
//                     the information relevant to this Analyze image (i.e. 
//                     dimensions, pixel spacing, etc.).
//
int AnalyzeReader::read( AnalyzeImage & di )
{
  if( file_ == "" ) 
  {
    cerr << "(AnalyzeReader::read) Error: No file selected." << endl;
    return -1;
  } 

  itk::AnalyzeImageIO::Pointer io = itk::AnalyzeImageIO::New();

  // Create a new reader
  typedef itk::ImageFileReader<ImageNDType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  // Set reader file name
  reader->SetFileName( const_cast<char*>(file_.c_str()) );

  reader->SetImageIO( io );

  try
  {
    reader->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cerr << ex;
    return -1;
  }

  ImageNDType::Pointer image = reader->GetOutput();

  di = AnalyzeImage( io, image, file_ );

  return 0;
}

} // End namespace SCIRun

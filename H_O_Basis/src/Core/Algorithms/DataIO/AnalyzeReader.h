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
 * HEADER (H) FILE : AnalyzeReader.h
 *
 * DESCRIPTION     : Provides ability to read and get information about
 *                   a set of Analyze files.  The read function can read
 *                   one valid Analyze series at a time.  The other functions
 *                   return information about what Analyze series' are in a 
 *                   directory and what files are in each series.  The user
 *                   can use that information to decide which directory/files
 *                   to read from.  The read function stores each Analyze 
 *                   series as a AnalyzeImage object, which in turn contains 
 *                   information about the series (dimensions, pixel spacing, 
 *                   etc.).
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

#ifndef AnalyzeReader_h
#define AnalyzeReader_h

// SCIRun includes

// Itk includes
#include "itkAnalyzeImageIOFactory.h"
#include "itkAnalyzeImageIO.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileReader.h"
#include <Core/Algorithms/DataIO/AnalyzeImage.h>

// Standard lib includes
#include <iostream>

namespace SCIRun {

// ****************************************************************************
// *************************** Class: AnalyzeReader ***************************
// ****************************************************************************

class AnalyzeReader
{

public:
  AnalyzeReader();
  ~AnalyzeReader();

  //! Reading functions
  void set_file( std::string file );
  std::string get_file();
  int read( AnalyzeImage & di );

private:

  std::string file_;

};

} // End namespace SCIRun
 
#endif // AnalyzeReader_h




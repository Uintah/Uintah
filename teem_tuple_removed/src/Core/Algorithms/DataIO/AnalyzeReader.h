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




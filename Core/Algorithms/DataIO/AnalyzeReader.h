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

#ifndef AnalyzeReader_h
#define AnalyzeReader_h

// Itk includes
#include "itkImageSeriesReader.h"
#include <Core/Algorithms/DataIO/AnalyzeImage.h>

// Standard lib includes
#include <iostream>

namespace SCIRun {

// ****************************************************************************
// ************************ Class: AnalyzeReader **************************
// ****************************************************************************

typedef itk::ImageSeriesReader<ImageNDType> ReaderType;

class AnalyzeReader
{

public:
  AnalyzeReader();
  ~AnalyzeReader();

  //! Reading function
  AnalyzeImage read_series( char * dir );

private:

protected:

};

} // End namespace SCIRun
 
#endif // AnalyzeReader_h




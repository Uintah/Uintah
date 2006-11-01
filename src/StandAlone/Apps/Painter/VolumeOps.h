//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : VolumeOps.h
//    Author : McKay Davis
//    Date   : Tue Oct 31 17:41:48 2006 (spooky!)

#ifndef LeXoV_VolumeOps_H
#define LeXoV_VolumeOps_H

#include <sci_defs/insight_defs.h>
#ifdef HAVE_INSIGHT

#include <Core/Datatypes/NrrdData.h>
#include <Core/Datatypes/ITKDatatype.h>
#include <Core/Util/Assert.h>
#include <itkImageToImageFilter.h>
#include <itkImportImageFilter.h>

namespace SCIRun {

class VolumeOps {
public:
  static unsigned int           nrrd_elem_count(NrrdDataHandle &);
  static unsigned int           nrrd_data_size(NrrdDataHandle &);
  static unsigned int           nrrd_type_size(NrrdDataHandle &);

  static NrrdDataHandle         float_to_bit(NrrdDataHandle &,
                                             float ref = 0.0,
                                             unsigned int mask = 0);


  static NrrdDataHandle         bit_to_float(NrrdDataHandle &,
                                             unsigned int mask = 0,
                                             float ref = 0.0);
                                            


  static NrrdDataHandle         clear_nrrd(NrrdDataHandle &);
  static NrrdDataHandle         create_clear_nrrd(NrrdDataHandle &, 
                                                  unsigned int type = 0);
  static NrrdDataHandle         create_nrrd(NrrdDataHandle &, 
                                            unsigned int type = 0);

};

}
#endif
#endif

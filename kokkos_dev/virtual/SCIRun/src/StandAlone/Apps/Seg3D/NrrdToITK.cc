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
//    File   : NrrdToITK.cc
//    Author : McKay Davis
//    Date   : Sun Oct 22 23:09:46 2006

#include <StandAlone/Apps/Seg3D/NrrdToITK.h>
#include <Core/Containers/StringUtil.h>

#ifdef HAVE_INSIGHT

namespace SCIRun {


ITKDatatypeHandle
nrrd_to_itk_image(NrrdDataHandle &nrrd) {
  Nrrd *n = nrrd->nrrd_;

  itk::Object::Pointer data = 0;
  switch (n->type) {
  case nrrdTypeChar: data = nrrd_to_itk_image<signed char>(n); break;
  case nrrdTypeUChar: data = nrrd_to_itk_image<unsigned char>(n); break;
  case nrrdTypeShort: data = nrrd_to_itk_image<signed short>(n); break;
  case nrrdTypeUShort: data = nrrd_to_itk_image<unsigned short>(n); break;
  case nrrdTypeInt: data = nrrd_to_itk_image<signed int>(n); break;
  case nrrdTypeUInt: data = nrrd_to_itk_image<unsigned int>(n); break;
  case nrrdTypeLLong: data = nrrd_to_itk_image<signed long long>(n); break;
  case nrrdTypeULLong: data =nrrd_to_itk_image<unsigned long long>(n); break;
  case nrrdTypeFloat: data = nrrd_to_itk_image<float>(n); break;
  case nrrdTypeDouble: data = nrrd_to_itk_image<double>(n); break;
  default: throw "nrrd_to_itk_image, cannot convert type" + to_string(n->type);
  }

  ITKDatatype *result = new SCIRun::ITKDatatype();
  result->data_ = data;
  return result;
}

}

#endif /* HAVE_INSIGHT */

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
 *  NotFinished.h:  Consistent way to keep track of holes in the code...
 *
 *  Written by:
 *   Wayne witzel
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCI_Core_Util_SizeTypeConvert_h
#define SCI_Core_Util_SizeTypeConvert_h

#include <sci_config.h>
#if HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#if HAVE_STDINT_H
#include <stdint.h>
#endif

namespace SCIRun{

  // pass in a pointer to a 64-bit int, but depending upon nByteMode it may
  // be treated as a 32-bit int (the last half wouldn't get touched).
  unsigned long convertSizeType(uint64_t* ssize, bool swapBytes,
				int nByteMode);    

} //end namespace SCIRun
#endif

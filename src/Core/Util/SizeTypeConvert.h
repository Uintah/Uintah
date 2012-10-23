/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */

#ifndef SCI_Core_Util_SizeTypeConvert_h
#define SCI_Core_Util_SizeTypeConvert_h

#include <sci_defs/config_defs.h>

#if HAVE_INTTYPES_H
#  include <inttypes.h>
#endif

#ifdef _WIN32
typedef unsigned long long uint64_t;
#endif


#include <Core/Util/share.h>

namespace SCIRun{

  // pass in a pointer to a 64-bit int, but depending upon nByteMode it may
  // be treated as a 32-bit int (the last half wouldn't get touched).
  SCISHARE unsigned long convertSizeType(uint64_t* ssize, bool swapBytes,
				int nByteMode);    

} //end namespace SCIRun
#endif

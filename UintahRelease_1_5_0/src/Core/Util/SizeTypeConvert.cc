/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#include <Core/Util/SizeTypeConvert.h>

#include <Core/Util/Endian.h>
#include <Core/Exceptions/InternalError.h>

namespace SCIRun {

unsigned long convertSizeType(uint64_t* ssize, bool swapBytes, int nByteMode)
{
  if (nByteMode == 4) {
    uint32_t size32 = *(uint32_t*)ssize;
    if (swapBytes) swapbytes(size32);
    return (unsigned long)size32;
  }
  else if (nByteMode == 8) {
    uint64_t size64 = *(uint64_t*)ssize;
    if (swapBytes) swapbytes(size64);

    if (sizeof(unsigned long) < 8 && size64 > 0xffffffff)
	throw InternalError("Overflow on 64 to 32 bit conversion", __FILE__, __LINE__);
  
    return (unsigned long)size64;
  }
  else {
    throw InternalError("Must be 32 or 64 bits", __FILE__, __LINE__);
  }
}

}


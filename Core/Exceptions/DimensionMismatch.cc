/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  DimensionMismatch.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 */

#include <Core/Exceptions/DimensionMismatch.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

namespace SCIRun {

DimensionMismatch::DimensionMismatch(long value, long expected, const char* file, int line)
    : value(value), expected(expected)
{
  std::ostringstream s;
  s << "A DimensionMismatch exception was thrown\n"
    << file << ":" << line << "\n"
    << "Dimension mismatch, got " << value << ", expected " << expected;
  msg = (char*)(s.str().c_str());
    
#ifdef EXCEPTIONS_CRASH
    std::cout << msg << "\n";
#endif
}

DimensionMismatch::DimensionMismatch(const DimensionMismatch& copy)
    : msg(strdup(copy.msg))
{
}
    
DimensionMismatch::~DimensionMismatch()
{
    free(msg);
}

const char* DimensionMismatch::message() const
{
    return msg;
}

const char* DimensionMismatch::type() const
{
    return "DimensionMismatch";
}
} // End namespace SCIRun

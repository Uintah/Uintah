/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
 *  ArrayIndexOutOfBounds.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 */

#include <Core/Exceptions/ArrayIndexOutOfBounds.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

namespace Uintah {

ArrayIndexOutOfBounds::ArrayIndexOutOfBounds(long value, long lower, long upper,
                                             const char* file, int line)
    : value(value), lower(lower), upper(upper)
{
  std::ostringstream s;
  s << "An ArrayIndexOutOfBounds exception was thrown.\n"
    << file << ":" << line << "\n"
    << "Array index " << value << " out of range ["
    << lower << ", " << upper << ")";
  msg = (char*)(s.str().c_str());
#ifdef EXCEPTIONS_CRASH
    std::cout << msg << "\n";
#endif
}

ArrayIndexOutOfBounds::ArrayIndexOutOfBounds(const ArrayIndexOutOfBounds& copy)
  : value(copy.value), lower(copy.lower), upper(copy.upper), msg(strdup(copy.msg))
{
}
    
ArrayIndexOutOfBounds::~ArrayIndexOutOfBounds()
{
}

const char* ArrayIndexOutOfBounds::message() const
{
    return msg;
}

const char* ArrayIndexOutOfBounds::type() const
{
    return "ArrayIndexOutOfBounds";
}

} // End namespace Uintah

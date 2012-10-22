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

/*
 *  InternalError.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 */

#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>

namespace SCIRun {


InternalError::InternalError(const std::string& message, const char* file, int line)
    : message_(message)
{
  std::ostringstream s;
  s << "An InternalError exception was thrown (" << message << ").\n"
    << "Error occurred in file: " << file << " on line " << line << ".\n";
  message_ = s.str();

#ifdef EXCEPTIONS_CRASH
  std::cout << message_ << "\n";
#endif
}

InternalError::InternalError(const InternalError& copy)
    : message_(copy.message_)
{
}

InternalError::~InternalError()
{
}

const char* InternalError::message() const
{
    return message_.c_str();
}

const char* InternalError::type() const
{
    return "InternalError";
}

} // End namespace SCIRun

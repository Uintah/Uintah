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

#include <Core/Exceptions/PapiInitializationError.h>
#include <iostream>
#include <sstream>

namespace SCIRun {


PapiInitializationError::PapiInitializationError(const std::string& message, const char* file, int line)
    : message_(message)
{
  std::ostringstream s;
  s << "An PapiInitializationError exception was thrown (" << message << ").\n"
    << "Error occurred in file: " << file << " on line " << line << ".\n";
  message_ = s.str();

#ifdef EXCEPTIONS_CRASH
  std::cout << message_ << "\n";
#endif
}

PapiInitializationError::PapiInitializationError(const PapiInitializationError& copy)
    : message_(copy.message_)
{
}

PapiInitializationError::~PapiInitializationError()
{
}

const char* PapiInitializationError::message() const
{
    return message_.c_str();
}

const char* PapiInitializationError::type() const
{
    return "PapiInitializationError";
}

} // End namespace SCIRun

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


#include <Core/Exceptions/TypeMismatchException.h>
#include <iostream>
#include <sstream>

using namespace Uintah;

TypeMismatchException::TypeMismatchException(const std::string& msg, const char* file, int line)
    : d_msg(msg)
{
  std::ostringstream s;
  s << "A TypeMismatchException was thrown.\n"
    << file << ":" << line << "\n" << d_msg;
  d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
  std::cout << d_msg << "\n";
#endif
}

TypeMismatchException::TypeMismatchException(const TypeMismatchException& copy)
    : d_msg(copy.d_msg)
{
}

TypeMismatchException::~TypeMismatchException()
{
}

const char* TypeMismatchException::message() const
{
    return d_msg.c_str();
}

const char* TypeMismatchException::type() const
{
    return "Uintah::Exceptions::TypeMismatchException";
}

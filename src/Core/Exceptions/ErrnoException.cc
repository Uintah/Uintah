/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
 *  ErrnoException.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 */

#include <Core/Exceptions/ErrnoException.h>

#include <sstream>
#include <cstring>
#include <iostream>

namespace SCIRun {

using namespace std;

ErrnoException::ErrnoException( const string & message, int err, const char* file, int line )
   : err_(err)
{
   ostringstream str;
   const char* s = strerror(err);
   if(!s)
      s="(none)";
   str << "An ErrnoException was thrown.\n"
       << file << ":" << line << "\n"
       << message << " (errno=" << err << ": " << s << ")";
   message_ = str.str();

#ifdef EXCEPTIONS_CRASH
   cout << message_ << "\n";
#endif
}

ErrnoException::ErrnoException(const ErrnoException& copy)
   : message_(copy.message_), err_(copy.err_)
{
}

ErrnoException::~ErrnoException()
{
}

const char* ErrnoException::message() const
{
   return message_.c_str();
}

const char* ErrnoException::type() const
{
   return "ErrnoException";
}

int ErrnoException::getErrno() const
{
   return err_;
}

} // End namespace SCIRun

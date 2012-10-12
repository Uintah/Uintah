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

#include <Core/Exceptions/InvalidCompressionMode.h>
#include <iostream>
#include <sstream>
using namespace Uintah;


InvalidCompressionMode::InvalidCompressionMode(const std::string& invalidmode,
					       const std::string& vartype,
                                               const char* file,
                                               int line)
{
  std::ostringstream s;
  s << "An InvalidCompressionMode exception was thrown.\n" 
    << file << ":" << line << "\n"
    << "'" << invalidmode << "' is not a valid compression mode";
  d_msg = s.str();

  if (vartype != "") {
    d_msg += std::string(" for a variable of type ") + vartype;
  }
  
#ifdef EXCEPTIONS_CRASH
  cout << d_msg << "\n";
#endif
}

InvalidCompressionMode::InvalidCompressionMode(const InvalidCompressionMode& copy)
    : d_msg(copy.d_msg)
{
}

InvalidCompressionMode::~InvalidCompressionMode()
{
}

const char* InvalidCompressionMode::message() const
{
  return d_msg.c_str();
}

const char* InvalidCompressionMode::type() const
{
  return "Packages/Uintah::Exceptions::InvalidCompressionMode";
}

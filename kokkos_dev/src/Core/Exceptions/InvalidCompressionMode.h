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
 *  InvalidCompressionMode.h: 
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   Febuary 2001
 *
 */

#ifndef UINTAH_EXCEPTIONS_INVALIDCOMPRESSIONMODE_H
#define UINTAH_EXCEPTIONS_INVALIDCOMPRESSIONMODE_H

#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {

  class InvalidCompressionMode : public SCIRun::Exception {
  public:
    InvalidCompressionMode(const std::string& invalidmode,
			   const std::string& vartype,
                           const char* file,
                           int line);
    InvalidCompressionMode(const InvalidCompressionMode&);
    virtual ~InvalidCompressionMode();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string d_msg;
    InvalidCompressionMode& operator=(const InvalidCompressionMode&);
  };
  
} // End namespace Uintah

#endif

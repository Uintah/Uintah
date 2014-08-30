/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef HLA_TestHelper_h
#define HLA_TestHelper_h

#include <iostream>

class TestHelper
{
  bool isokay;
  bool report;
public:
  TestHelper( const bool reportResults=true )
    : report( reportResults )
  {
    isokay=true;
  }

  void operator()( const bool result )
  {
    if( isokay )
      isokay = result;
    if( report ){
      if( result )
        std::cout << "  pass " << std::endl;
      else
        std::cout << "  fail " << std::endl;
    }
  }

  template<typename T>
  void operator()( const bool result, const T t )
  {
    if( isokay )
      isokay = result;
    if( report ){
      if( result )
        std::cout << "  pass " << t << std::endl;
      else
        std::cout << "  fail " << t << std::endl;
    }
  }

  bool ok() const{ return isokay; }

  bool isfailed() const{ return !isokay; }

};

#endif // HLA_TestHelper_h

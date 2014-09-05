/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  PingThrow_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Kosta Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   July 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef PingThrow_PingThrow_impl_h
#define PingThrow_PingThrow_impl_h

#include <testprograms/Component/exceptiontest/PingThrow_sidl.h>

namespace PingThrow_ns {

    class PingThrow_impl : public PingThrow {
    public:
      PingThrow_impl();
      virtual ~PingThrow_impl();
      virtual int pingthrow(int);
      virtual int pingthrow2(int);
      virtual void getOX(OtherThrow::pointer& otptr);
      virtual void donone();
    };

    class OtherThrow_impl : public OtherThrow {
    public:
      OtherThrow_impl();
      virtual ~OtherThrow_impl();
      virtual int otherthrow();
    };

} // End namespace pingthrow

#endif


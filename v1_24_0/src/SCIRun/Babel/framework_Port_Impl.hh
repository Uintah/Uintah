//
//  For more information, please see: http://software.sci.utah.edu
// 
//  The MIT License
// 
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
// 
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
// 
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
// 
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//

// 
// File:          framework_Port_Impl.hh
// Symbol:        framework.Port-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040129 15:00:03 MST
// Generated:     20040129 15:00:06 MST
// Description:   Server-side implementation for framework.Port
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 14
// source-url    = file:/home/sci/damevski/SCIRun/ccadebug-RH8/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_Port_Impl_hh
#define included_framework_Port_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_Port_IOR_h
#include "framework_Port_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_framework_Port_hh
#include "framework_Port.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.Port._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.Port._includes)

namespace framework { 

  /**
   * Symbol "framework.Port" (version 1.0)
   */
  class Port_impl
  // DO-NOT-DELETE splicer.begin(framework.Port._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.Port._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Port self;

    // DO-NOT-DELETE splicer.begin(framework.Port._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(framework.Port._implementation)

  private:
    // private default constructor (required)
    Port_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Port_impl( struct framework_Port__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Port_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

  };  // end class Port_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.Port._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.Port._misc)

#endif

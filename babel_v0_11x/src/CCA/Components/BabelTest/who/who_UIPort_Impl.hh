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
// File:          who_UIPort_Impl.hh
// Symbol:        who.UIPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030915 14:59:08 MST
// Generated:     20030915 14:59:12 MST
// Description:   Server-side implementation for who.UIPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 10
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 

#ifndef included_who_UIPort_Impl_hh
#define included_who_UIPort_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_who_UIPort_IOR_h
#include "who_UIPort_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_who_UIPort_hh
#include "who_UIPort.hh"
#endif


// DO-NOT-DELETE splicer.begin(who.UIPort._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(who.UIPort._includes)

namespace who { 

  /**
   * Symbol "who.UIPort" (version 1.0)
   */
  class UIPort_impl
  // DO-NOT-DELETE splicer.begin(who.UIPort._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(who.UIPort._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    UIPort self;

    // DO-NOT-DELETE splicer.begin(who.UIPort._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(who.UIPort._implementation)

  private:
    // private default constructor (required)
    UIPort_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    UIPort_impl( struct who_UIPort__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~UIPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

    /**
     * user defined non-static method.
     */
    int32_t
    ui() throw () 
    ;
  };  // end class UIPort_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.UIPort._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(who.UIPort._misc)

#endif

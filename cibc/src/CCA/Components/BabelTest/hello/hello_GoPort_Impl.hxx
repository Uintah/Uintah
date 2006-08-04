//
// For more information, please see: http://software.sci.utah.edu
//
// The MIT License
//
// Copyright (c) 2005 Scientific Computing and Imaging Institute,
// University of Utah.
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
// 
// File:          hello_GoPort_Impl.hxx
// Symbol:        hello.GoPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for hello.GoPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_hello_GoPort_Impl_hxx
#define included_hello_GoPort_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
#endif
#ifndef included_hello_GoPort_IOR_h
#include "hello_GoPort_IOR.h"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_gov_cca_ports_GoPort_hxx
#include "gov_cca_ports_GoPort.hxx"
#endif
#ifndef included_hello_GoPort_hxx
#include "hello_GoPort.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif


// DO-NOT-DELETE splicer.begin(hello.GoPort._includes)
// Insert-Code-Here {hello.GoPort._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(hello.GoPort._includes)

namespace hello { 

  /**
   * Symbol "hello.GoPort" (version 1.0)
   */
  class GoPort_impl : public virtual UCXX ::hello::GoPort 
  // DO-NOT-DELETE splicer.begin(hello.GoPort._inherits)
  // Insert-Code-Here {hello.GoPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(hello.GoPort._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(hello.GoPort._implementation)
    UCXX ::gov::cca::Services svc;
    // DO-NOT-DELETE splicer.end(hello.GoPort._implementation)

  public:
    // default constructor, shouldn't be used (required)
    GoPort_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      GoPort_impl( struct hello_GoPort__object * s ) : StubBase(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~GoPort_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:

      /**
       * user defined non-static method.
       */
      void
      setServices_impl (
        /* in */UCXX ::gov::cca::Services services
      )
      ;


      /**
       * Execute some encapsulated functionality on the component.
       * Return 0 if ok, -1 if internal error but component may be
       * used further, and -2 if error so severe that component cannot
       * be further used safely.
       */
      int32_t
      go_impl() ;
    };  // end class GoPort_impl

  } // end namespace hello

  // DO-NOT-DELETE splicer.begin(hello.GoPort._misc)
  // Insert-Code-Here {hello.GoPort._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(hello.GoPort._misc)

  #endif

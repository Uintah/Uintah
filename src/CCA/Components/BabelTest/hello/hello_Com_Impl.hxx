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
// File:          hello_Com_Impl.hxx
// Symbol:        hello.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for hello.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_hello_Com_Impl_hxx
#define included_hello_Com_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
#endif
#ifndef included_hello_Com_IOR_h
#include "hello_Com_IOR.h"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_hello_Com_hxx
#include "hello_Com.hxx"
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


// DO-NOT-DELETE splicer.begin(hello.Com._includes)
// Insert-Code-Here {hello.Com._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(hello.Com._includes)

namespace hello { 

  /**
   * Symbol "hello.Com" (version 1.0)
   */
  class Com_impl : public virtual UCXX ::hello::Com 
  // DO-NOT-DELETE splicer.begin(hello.Com._inherits)
  // Insert-Code-Here {hello.Com._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(hello.Com._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(hello.Com._implementation)
    UCXX ::gov::cca::Services svc;
    // DO-NOT-DELETE splicer.end(hello.Com._implementation)

  public:
    // default constructor, shouldn't be used (required)
    Com_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      Com_impl( struct hello_Com__object * s ) : StubBase(s,true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~Com_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:


      /**
       * Starts up a component presence in the calling framework.
       * @param Svc the component instance's handle on the framework world.
       * Contracts concerning Svc and setServices:
       * 
       * The component interaction with the CCA framework
       * and Ports begins on the call to setServices by the framework.
       * 
       * This function is called exactly once for each instance created
       * by the framework.
       * 
       * The argument Svc will never be nil/null.
       * 
       * Those uses ports which are automatically connected by the framework
       * (so-called service-ports) may be obtained via getPort during
       * setServices.
       */
      void
      setServices_impl (
        /* in */UCXX ::gov::cca::Services services
      )
      ;

    };  // end class Com_impl

  } // end namespace hello

  // DO-NOT-DELETE splicer.begin(hello.Com._misc)
  // Insert-Code-Here {hello.Com._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(hello.Com._misc)

  #endif

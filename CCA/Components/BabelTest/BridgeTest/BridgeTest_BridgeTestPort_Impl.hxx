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
// File:          BridgeTest_BridgeTestPort_Impl.hxx
// Symbol:        BridgeTest.BridgeTestPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for BridgeTest.BridgeTestPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_BridgeTest_BridgeTestPort_Impl_hxx
#define included_BridgeTest_BridgeTestPort_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
#endif
#ifndef included_BridgeTest_BridgeTestPort_IOR_h
#include "BridgeTest_BridgeTestPort_IOR.h"
#endif
#ifndef included_BridgeTest_BridgeTestPort_hxx
#include "BridgeTest_BridgeTestPort.hxx"
#endif
#ifndef included_BridgeTest_iBridgeTestPort_hxx
#include "BridgeTest_iBridgeTestPort.hxx"
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


// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._includes)
// Insert-Code-Here {BridgeTest.BridgeTestPort._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._includes)

namespace BridgeTest { 

  /**
   * Symbol "BridgeTest.BridgeTestPort" (version 1.0)
   */
  class BridgeTestPort_impl : public virtual UCXX ::BridgeTest::BridgeTestPort 
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._inherits)
  // Insert-Code-Here {BridgeTest.BridgeTestPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._implementation)
    // Insert-Code-Here {BridgeTest.BridgeTestPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._implementation)

  public:
    // default constructor, shouldn't be used (required)
    BridgeTestPort_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      BridgeTestPort_impl( struct BridgeTest_BridgeTestPort__object * s ) : 
        StubBase(s,true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~BridgeTestPort_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:

      /**
       * user defined non-static method.
       */
      void
      m2_impl (
        /* in array<int> */UCXX ::sidl::array<int32_t> a
      )
      ;

    };  // end class BridgeTestPort_impl

  } // end namespace BridgeTest

// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._misc)
// Insert-Code-Here {BridgeTest.BridgeTestPort._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._misc)

  #endif

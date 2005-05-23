// 
// File:          BridgeTest_BridgeTestPort_Impl.hh
// Symbol:        BridgeTest.BridgeTestPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040412 13:28:09 MST
// Generated:     20040412 13:28:11 MST
// Description:   Server-side implementation for BridgeTest.BridgeTestPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 15
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/BridgeTest/BridgeTest.sidl
// 

#ifndef included_BridgeTest_BridgeTestPort_Impl_hh
#define included_BridgeTest_BridgeTestPort_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_BridgeTest_BridgeTestPort_IOR_h
#include "BridgeTest_BridgeTestPort_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_BridgeTest_BridgeTestPort_hh
#include "BridgeTest_BridgeTestPort.hh"
#endif
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif


// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._includes)

namespace BridgeTest { 

  /**
   * Symbol "BridgeTest.BridgeTestPort" (version 1.0)
   */
  class BridgeTestPort_impl
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    BridgeTestPort self;

    // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._implementation)

  private:
    // private default constructor (required)
    BridgeTestPort_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    BridgeTestPort_impl( struct BridgeTest_BridgeTestPort__object * s ) : 
      self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~BridgeTestPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

    /**
     * user defined non-static method.
     */
    void
    m2 (
      /*in*/ ::SIDL::array<int> a
    )
    throw () 
    ;

  };  // end class BridgeTestPort_impl

} // end namespace BridgeTest

// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._misc)

#endif

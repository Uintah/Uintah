// 
// File:          BridgeTest_BridgeTestPort_Impl.hh
// Symbol:        BridgeTest.BridgeTestPort-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for BridgeTest.BridgeTestPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 

#ifndef included_BridgeTest_BridgeTestPort_Impl_hh
#define included_BridgeTest_BridgeTestPort_Impl_hh

#ifndef included_sidl_cxx_hh
#include "sidl_cxx.hh"
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
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._includes)
// Insert-Code-Here {BridgeTest.BridgeTestPort._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._includes)

namespace BridgeTest { 

  /**
   * Symbol "BridgeTest.BridgeTestPort" (version 1.0)
   */
  class BridgeTestPort_impl
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._inherits)
  // Insert-Code-Here {BridgeTest.BridgeTestPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    BridgeTestPort self;

    // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._implementation)
    // Insert-Code-Here {BridgeTest.BridgeTestPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._implementation)

  private:
    // private default constructor (required)
    BridgeTestPort_impl() 
    {} 

  public:
    // sidl constructor (required)
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

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    m2 (
      /* in */ ::sidl::array<int32_t> a
    )
    throw () 
    ;

  };  // end class BridgeTestPort_impl

} // end namespace BridgeTest

// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._misc)
// Insert-Code-Here {BridgeTest.BridgeTestPort._misc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._misc)

#endif

// 
// File:          BridgeTest_Com_Impl.hh
// Symbol:        BridgeTest.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040412 13:28:09 MST
// Generated:     20040412 13:28:11 MST
// Description:   Server-side implementation for BridgeTest.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 18
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/BridgeTest/BridgeTest.sidl
// 

#ifndef included_BridgeTest_Com_Impl_hh
#define included_BridgeTest_Com_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_BridgeTest_Com_IOR_h
#include "BridgeTest_Com_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_BridgeTest_Com_hh
#include "BridgeTest_Com.hh"
#endif
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_gov_cca_Services_hh
#include "gov_cca_Services.hh"
#endif


// DO-NOT-DELETE splicer.begin(BridgeTest.Com._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(BridgeTest.Com._includes)

namespace BridgeTest { 

  /**
   * Symbol "BridgeTest.Com" (version 1.0)
   */
  class Com_impl
  // DO-NOT-DELETE splicer.begin(BridgeTest.Com._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(BridgeTest.Com._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Com self;

    // DO-NOT-DELETE splicer.begin(BridgeTest.Com._implementation)
    gov::cca::Services svc;
    // DO-NOT-DELETE splicer.end(BridgeTest.Com._implementation)

  private:
    // private default constructor (required)
    Com_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Com_impl( struct BridgeTest_Com__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Com_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:


    /**
     * Obtain Services handle, through which the 
     * component communicates with the framework. 
     * This is the one method that every CCA Component
     * must implement. 
     */
    void
    setServices (
      /*in*/ ::gov::cca::Services services
    )
    throw () 
    ;

  };  // end class Com_impl

} // end namespace BridgeTest

// DO-NOT-DELETE splicer.begin(BridgeTest.Com._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(BridgeTest.Com._misc)

#endif

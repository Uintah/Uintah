// 
// File:          BridgeTest_BridgeTestPort_Impl.hxx
// Symbol:        BridgeTest.BridgeTestPort-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for BridgeTest.BridgeTestPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_BridgeTest_BridgeTestPort_Impl_hxx
#define included_BridgeTest_BridgeTestPort_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
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


// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._hincludes)
// insert code here (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._hincludes)

namespace BridgeTest { 

  /**
   * Symbol "BridgeTest.BridgeTestPort" (version 1.0)
   */
  class BridgeTestPort_impl : public virtual ::BridgeTest::BridgeTestPort 
  // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._inherits)
  // Insert-Code-Here {BridgeTest.BridgeTestPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._implementation)
    // Insert-Code-Here {BridgeTest.BridgeTestPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._implementation)

  public:
    // default constructor, used for data wrapping(required)
    BridgeTestPort_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      BridgeTestPort_impl( struct BridgeTest_BridgeTestPort__object * ior ) : 
        StubBase(ior,true), 
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::BridgeTest::iBridgeTestPort((ior==NULL) ? NULL : &((
      *ior).d_bridgetest_ibridgetestport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~BridgeTestPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:

    /**
     * user defined non-static method.
     */
    void
    m2_impl (
      /* in array<int> */::sidl::array<int32_t>& a
    )
    ;

  };  // end class BridgeTestPort_impl

} // end namespace BridgeTest

// DO-NOT-DELETE splicer.begin(BridgeTest.BridgeTestPort._hmisc)
// insert code here (miscellaneous things)
// DO-NOT-DELETE splicer.end(BridgeTest.BridgeTestPort._hmisc)

#endif

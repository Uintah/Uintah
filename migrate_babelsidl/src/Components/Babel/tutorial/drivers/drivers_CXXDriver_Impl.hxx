// 
// File:          drivers_CXXDriver_Impl.hxx
// Symbol:        drivers.CXXDriver-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for drivers.CXXDriver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_drivers_CXXDriver_Impl_hxx
#define included_drivers_CXXDriver_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_drivers_CXXDriver_IOR_h
#include "drivers_CXXDriver_IOR.h"
#endif
#ifndef included_drivers_CXXDriver_hxx
#include "drivers_CXXDriver.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_gov_cca_ports_GoPort_hxx
#include "gov_cca_ports_GoPort.hxx"
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
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif


// DO-NOT-DELETE splicer.begin(drivers.CXXDriver._hincludes)
#include "integrator_IntegratorPort.hxx"
#include "gov_cca_ports_GoPort.hxx"
// DO-NOT-DELETE splicer.end(drivers.CXXDriver._hincludes)

namespace drivers { 

  /**
   * Symbol "drivers.CXXDriver" (version 1.0)
   */
  class CXXDriver_impl : public virtual ::drivers::CXXDriver 
  // DO-NOT-DELETE splicer.begin(drivers.CXXDriver._inherits)
  // Insert-Code-Here {drivers.CXXDriver._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(drivers.CXXDriver._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(drivers.CXXDriver._implementation)
    // Insert-Code-Here {drivers.CXXDriver._implementation} (additional details)

    ::gov::cca::Services frameworkServices;

    // DO-NOT-DELETE splicer.end(drivers.CXXDriver._implementation)

  public:
    // default constructor, used for data wrapping(required)
    CXXDriver_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      CXXDriver_impl( struct drivers_CXXDriver__object * ior ) : StubBase(ior,
        true), 
      ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::gov::cca::ports::GoPort((ior==NULL) ? NULL : &((
      *ior).d_gov_cca_ports_goport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~CXXDriver_impl() { _dtor(); }

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
    int32_t
    go_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setServices_impl (
      /* in */::gov::cca::Services& services
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

  };  // end class CXXDriver_impl

} // end namespace drivers

// DO-NOT-DELETE splicer.begin(drivers.CXXDriver._hmisc)
// Insert-Code-Here {drivers.CXXDriver._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(drivers.CXXDriver._hmisc)

#endif

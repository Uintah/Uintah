// 
// File:          framework_Component_Impl.hxx
// Symbol:        framework.Component-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for framework.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_framework_Component_Impl_hxx
#define included_framework_Component_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_framework_Component_IOR_h
#include "framework_Component_IOR.h"
#endif
#ifndef included_framework_Component_hxx
#include "framework_Component.hxx"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
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


// DO-NOT-DELETE splicer.begin(framework.Component._hincludes)
// insert code here (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(framework.Component._hincludes)

namespace framework { 

  /**
   * Symbol "framework.Component" (version 1.0)
   */
  class Component_impl : public virtual ::framework::Component 
  // DO-NOT-DELETE splicer.begin(framework.Component._inherits)
  // Insert-Code-Here {framework.Component._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(framework.Component._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(framework.Component._implementation)
    // Insert-Code-Here {framework.Component._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(framework.Component._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Component_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      Component_impl( struct framework_Component__object * ior ) : StubBase(ior,
        true), 
    ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)) , 
      _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Component_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


    /**
     *  Starts up a component presence in the calling framework.
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
      /* in */::gov::cca::Services& services
    )
    ;

  };  // end class Component_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.Component._hmisc)
// insert code here (miscellaneous things)
// DO-NOT-DELETE splicer.end(framework.Component._hmisc)

#endif

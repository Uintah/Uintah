// 
// File:          framework_Component_Impl.hxx
// Symbol:        framework.Component-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for framework.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_framework_Component_Impl_hxx
#define included_framework_Component_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
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


// DO-NOT-DELETE splicer.begin(framework.Component._includes)
// Insert-Code-Here {framework.Component._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(framework.Component._includes)

namespace framework { 

  /**
   * Symbol "framework.Component" (version 1.0)
   */
  class Component_impl : public virtual UCXX ::framework::Component 
  // DO-NOT-DELETE splicer.begin(framework.Component._inherits)
  // Insert-Code-Here {framework.Component._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(framework.Component._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(framework.Component._implementation)
    // Insert-Code-Here {framework.Component._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(framework.Component._implementation)

  public:
    // default constructor, shouldn't be used (required)
    Component_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      Component_impl( struct framework_Component__object * s ) : StubBase(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~Component_impl() { _dtor(); }

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

    };  // end class Component_impl

  } // end namespace framework

  // DO-NOT-DELETE splicer.begin(framework.Component._misc)
  // Insert-Code-Here {framework.Component._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(framework.Component._misc)

  #endif

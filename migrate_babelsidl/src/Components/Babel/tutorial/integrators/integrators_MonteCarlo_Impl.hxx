// 
// File:          integrators_MonteCarlo_Impl.hxx
// Symbol:        integrators.MonteCarlo-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for integrators.MonteCarlo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_integrators_MonteCarlo_Impl_hxx
#define included_integrators_MonteCarlo_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_integrators_MonteCarlo_IOR_h
#include "integrators_MonteCarlo_IOR.h"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_ComponentRelease_hxx
#include "gov_cca_ComponentRelease.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_integrator_IntegratorPort_hxx
#include "integrator_IntegratorPort.hxx"
#endif
#ifndef included_integrators_MonteCarlo_hxx
#include "integrators_MonteCarlo.hxx"
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


// DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._hincludes)
#include "integrator_IntegratorPort.hxx"
#include "function_FunctionPort.hxx"
#include "randomgen_RandomGeneratorPort.hxx"
// DO-NOT-DELETE splicer.end(integrators.MonteCarlo._hincludes)

namespace integrators { 

  /**
   * Symbol "integrators.MonteCarlo" (version 1.0)
   */
  class MonteCarlo_impl : public virtual ::integrators::MonteCarlo 
  // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._inherits)
  // Insert-Code-Here {integrators.MonteCarlo._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(integrators.MonteCarlo._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._implementation)
    ::gov::cca::Services frameworkServices;
    // DO-NOT-DELETE splicer.end(integrators.MonteCarlo._implementation)

  public:
    // default constructor, used for data wrapping(required)
    MonteCarlo_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      MonteCarlo_impl( struct integrators_MonteCarlo__object * ior ) : StubBase(
        ior,true), 
      ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
      ::gov::cca::ComponentRelease((ior==NULL) ? NULL : &((
        *ior).d_gov_cca_componentrelease)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::integrator::IntegratorPort((ior==NULL) ? NULL : &((
      *ior).d_integrator_integratorport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~MonteCarlo_impl() { _dtor(); }

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
    double
    integrate_impl (
      /* in */double lowBound,
      /* in */double upBound,
      /* in */int32_t count
    )
    ;

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

    /**
     * user defined non-static method.
     */
    void
    releaseServices_impl (
      /* in */::gov::cca::Services& services
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

  };  // end class MonteCarlo_impl

} // end namespace integrators

// DO-NOT-DELETE splicer.begin(integrators.MonteCarlo._hmisc)
// Insert-Code-Here {integrators.MonteCarlo._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(integrators.MonteCarlo._hmisc)

#endif

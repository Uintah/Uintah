// 
// File:          randomgens_RandNumGenerator_Impl.hxx
// Symbol:        randomgens.RandNumGenerator-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for randomgens.RandNumGenerator
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_randomgens_RandNumGenerator_Impl_hxx
#define included_randomgens_RandNumGenerator_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_randomgens_RandNumGenerator_IOR_h
#include "randomgens_RandNumGenerator_IOR.h"
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
#ifndef included_randomgen_RandomGeneratorPort_hxx
#include "randomgen_RandomGeneratorPort.hxx"
#endif
#ifndef included_randomgens_RandNumGenerator_hxx
#include "randomgens_RandNumGenerator.hxx"
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


// DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._hincludes)
// Insert-Code-Here {randomgens.RandNumGenerator._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._hincludes)

namespace randomgens { 

  /**
   * Symbol "randomgens.RandNumGenerator" (version 1.0)
   */
  class RandNumGenerator_impl : public virtual ::randomgens::RandNumGenerator 
  // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._inherits)
  // Insert-Code-Here {randomgens.RandNumGenerator._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._implementation)
    // Insert-Code-Here {randomgens.RandNumGenerator._implementation} (additional details)
    gov::cca::Services frameworkServices;
    // DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._implementation)

  public:
    // default constructor, used for data wrapping(required)
    RandNumGenerator_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      RandNumGenerator_impl( struct randomgens_RandNumGenerator__object * ior ) 
        : StubBase(ior,true), 
      ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::randomgen::RandomGeneratorPort((ior==NULL) ? NULL : &((
      *ior).d_randomgen_randomgeneratorport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~RandNumGenerator_impl() { _dtor(); }

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
    getRandomNumber_impl() ;

    /**
     *  Starts up a component presence in the calling framework.
     * @param services the component instance's handle on the framework world.
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
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

  };  // end class RandNumGenerator_impl

} // end namespace randomgens

// DO-NOT-DELETE splicer.begin(randomgens.RandNumGenerator._hmisc)
// Insert-Code-Here {randomgens.RandNumGenerator._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(randomgens.RandNumGenerator._hmisc)

#endif

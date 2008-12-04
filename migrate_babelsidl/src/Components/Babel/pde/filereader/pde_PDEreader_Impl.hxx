// 
// File:          pde_PDEreader_Impl.hxx
// Symbol:        pde.PDEreader-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.PDEreader
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_pde_PDEreader_Impl_hxx
#define included_pde_PDEreader_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_pde_PDEreader_IOR_h
#include "pde_PDEreader_IOR.h"
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
#ifndef included_pde_PDEreader_hxx
#include "pde_PDEreader.hxx"
#endif
#ifndef included_pdeports_PDEdescriptionPort_hxx
#include "pdeports_PDEdescriptionPort.hxx"
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


// DO-NOT-DELETE splicer.begin(pde.PDEreader._hincludes)
// Insert-Code-Here {pde.PDEreader._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(pde.PDEreader._hincludes)

namespace pde { 

  /**
   * Symbol "pde.PDEreader" (version 0.1)
   */
  class PDEreader_impl : public virtual ::pde::PDEreader 
  // DO-NOT-DELETE splicer.begin(pde.PDEreader._inherits)
  // Insert-Code-Here {pde.PDEreader._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(pde.PDEreader._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(pde.PDEreader._implementation)
    // Insert-Code-Here {pde.PDEreader._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(pde.PDEreader._implementation)

  public:
    // default constructor, used for data wrapping(required)
    PDEreader_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      PDEreader_impl( struct pde_PDEreader__object * ior ) : StubBase(ior,true),
        
      ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::pdeports::PDEdescriptionPort((ior==NULL) ? NULL : &((
      *ior).d_pdeports_pdedescriptionport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~PDEreader_impl() { _dtor(); }

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
    getPDEdescription_impl (
      /* out array<double> */::sidl::array<double>& nodes,
      /* out array<int> */::sidl::array<int32_t>& boundaries,
      /* out array<int> */::sidl::array<int32_t>& dirichletNodes,
      /* out array<double> */::sidl::array<double>& dirichletValues
    )
    ;


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
    //    ::gov::cca::CCAException
    //    ::sidl::RuntimeException
    ;

  };  // end class PDEreader_impl

} // end namespace pde

// DO-NOT-DELETE splicer.begin(pde.PDEreader._hmisc)
// Insert-Code-Here {pde.PDEreader._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(pde.PDEreader._hmisc)

#endif

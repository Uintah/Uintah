// 
// File:          pde_FEM_Impl.hxx
// Symbol:        pde.FEM-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.FEM
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_pde_FEM_Impl_hxx
#define included_pde_FEM_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_pde_FEM_IOR_h
#include "pde_FEM_IOR.h"
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
#ifndef included_pde_FEM_hxx
#include "pde_FEM.hxx"
#endif
#ifndef included_pdeports_FEMmatrixPort_hxx
#include "pdeports_FEMmatrixPort.hxx"
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


// DO-NOT-DELETE splicer.begin(pde.FEM._hincludes)
// Insert-Code-Here {pde.FEM._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(pde.FEM._hincludes)

namespace pde { 

  /**
   * Symbol "pde.FEM" (version 0.1)
   */
  class FEM_impl : public virtual ::pde::FEM 
  // DO-NOT-DELETE splicer.begin(pde.FEM._inherits)
  // Insert-Code-Here {pde.FEM._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(pde.FEM._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(pde.FEM._implementation)
    // Insert-Code-Here {pde.FEM._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(pde.FEM._implementation)

  public:
    // default constructor, used for data wrapping(required)
    FEM_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      FEM_impl( struct pde_FEM__object * ior ) : StubBase(ior,true), 
      ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::pdeports::FEMmatrixPort((ior==NULL) ? NULL : &((
      *ior).d_pdeports_femmatrixport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~FEM_impl() { _dtor(); }

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
    makeFEMmatrices_impl (
      /* in array<int> */::sidl::array<int32_t>& mesh,
      /* in array<double> */::sidl::array<double>& nodes,
      /* in array<int> */::sidl::array<int32_t>& dirichletNodes,
      /* in array<double> */::sidl::array<double>& dirichletValues,
      /* out array<double,2> */::sidl::array<double>& Ag,
      /* out array<double> */::sidl::array<double>& fg,
      /* out */int32_t& size
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

  };  // end class FEM_impl

} // end namespace pde

// DO-NOT-DELETE splicer.begin(pde.FEM._hmisc)
// Insert-Code-Here {pde.FEM._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(pde.FEM._hmisc)

#endif

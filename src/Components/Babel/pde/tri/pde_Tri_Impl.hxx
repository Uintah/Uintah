// 
// File:          pde_Tri_Impl.hxx
// Symbol:        pde.Tri-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.Tri
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_pde_Tri_Impl_hxx
#define included_pde_Tri_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_pde_Tri_IOR_h
#include "pde_Tri_IOR.h"
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
#ifndef included_pde_Tri_hxx
#include "pde_Tri.hxx"
#endif
#ifndef included_pdeports_MeshPort_hxx
#include "pdeports_MeshPort.hxx"
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


// DO-NOT-DELETE splicer.begin(pde.Tri._hincludes)
// Insert-Code-Here {pde.Tri._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(pde.Tri._hincludes)

namespace pde { 

  /**
   * Symbol "pde.Tri" (version 0.1)
   */
  class Tri_impl : public virtual ::pde::Tri 
  // DO-NOT-DELETE splicer.begin(pde.Tri._inherits)
  // Insert-Code-Here {pde.Tri._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(pde.Tri._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(pde.Tri._implementation)
    // Insert-Code-Here {pde.Tri._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(pde.Tri._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Tri_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      Tri_impl( struct pde_Tri__object * ior ) : StubBase(ior,true), 
      ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::pdeports::MeshPort((ior==NULL) ? NULL : &((*ior).d_pdeports_meshport)) , 
      _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Tri_impl() { _dtor(); }

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
    triangulate_impl (
      /* in array<double> */::sidl::array<double>& nodes,
      /* in array<int> */::sidl::array<int32_t>& boundaries,
      /* out array<int> */::sidl::array<int32_t>& triangles
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

  };  // end class Tri_impl

} // end namespace pde

// DO-NOT-DELETE splicer.begin(pde.Tri._hmisc)
// Insert-Code-Here {pde.Tri._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(pde.Tri._hmisc)

#endif

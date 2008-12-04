// 
// File:          pde_PDEdriver_Impl.hxx
// Symbol:        pde.PDEdriver-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for pde.PDEdriver
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_pde_PDEdriver_Impl_hxx
#define included_pde_PDEdriver_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_pde_PDEdriver_IOR_h
#include "pde_PDEdriver_IOR.h"
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
#ifndef included_pde_PDEdriver_hxx
#include "pde_PDEdriver.hxx"
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


// DO-NOT-DELETE splicer.begin(pde.PDEdriver._hincludes)
#include <pdeports.hxx>
// DO-NOT-DELETE splicer.end(pde.PDEdriver._hincludes)

namespace pde { 

  /**
   * Symbol "pde.PDEdriver" (version 0.1)
   */
  class PDEdriver_impl : public virtual ::pde::PDEdriver 
  // DO-NOT-DELETE splicer.begin(pde.PDEdriver._inherits)
  // Insert-Code-Here {pde.PDEdriver._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(pde.PDEdriver._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(pde.PDEdriver._implementation)
    gov::cca::Services services;
    // DO-NOT-DELETE splicer.end(pde.PDEdriver._implementation)

  public:
    // default constructor, used for data wrapping(required)
    PDEdriver_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      PDEdriver_impl( struct pde_PDEdriver__object * ior ) : StubBase(ior,true),
        
      ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::gov::cca::ports::GoPort((ior==NULL) ? NULL : &((
      *ior).d_gov_cca_ports_goport)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~PDEdriver_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


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


    /**
     *  
     * Execute some encapsulated functionality on the component. 
     * Return 0 if ok, -1 if internal error but component may be 
     * used further, and -2 if error so severe that component cannot
     * be further used safely.
     */
    int32_t
    go_impl() ;
  };  // end class PDEdriver_impl

} // end namespace pde

// DO-NOT-DELETE splicer.begin(pde.PDEdriver._hmisc)
// Insert-Code-Here {pde.PDEdriver._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(pde.PDEdriver._hmisc)

#endif

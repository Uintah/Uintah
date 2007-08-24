// 
// File:          HelloServer_Component_Impl.hxx
// Symbol:        HelloServer.Component-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for HelloServer.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_HelloServer_Component_Impl_hxx
#define included_HelloServer_Component_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_HelloServer_Component_IOR_h
#include "HelloServer_Component_IOR.h"
#endif
#ifndef included_HelloServer_Component_hxx
#include "HelloServer_Component.hxx"
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


// DO-NOT-DELETE splicer.begin(HelloServer.Component._hincludes)
// Insert-Code-Here {HelloServer.Component._hincludes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(HelloServer.Component._hincludes)

namespace HelloServer { 

  /**
   * Symbol "HelloServer.Component" (version 1.0)
   */
  class Component_impl : public virtual ::HelloServer::Component 
  // DO-NOT-DELETE splicer.begin(HelloServer.Component._inherits)
  // Insert-Code-Here {HelloServer.Component._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(HelloServer.Component._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(HelloServer.Component._implementation)
    // Insert-Code-Here {HelloServer.Component._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(HelloServer.Component._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Component_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      Component_impl( struct HelloServer_Component__object * ior ) : StubBase(
        ior,true), 
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
     * Method <code>setServices</code> is called by the framework.
     */
    void
    setServices_impl (
      /* in */::gov::cca::Services& services
    )
    // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

  };  // end class Component_impl

} // end namespace HelloServer

// DO-NOT-DELETE splicer.begin(HelloServer.Component._hmisc)
// Insert-Code-Here {HelloServer.Component._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(HelloServer.Component._hmisc)

#endif

// 
// File:          framework_ComponentID_Impl.hxx
// Symbol:        framework.ComponentID-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for framework.ComponentID
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_framework_ComponentID_Impl_hxx
#define included_framework_ComponentID_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_framework_ComponentID_IOR_h
#include "framework_ComponentID_IOR.h"
#endif
#ifndef included_framework_ComponentID_hxx
#include "framework_ComponentID.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
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


// DO-NOT-DELETE splicer.begin(framework.ComponentID._hincludes)
// insert code here (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(framework.ComponentID._hincludes)

namespace framework { 

  /**
   * Symbol "framework.ComponentID" (version 1.0)
   */
  class ComponentID_impl : public virtual ::framework::ComponentID 
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._inherits)
  // Insert-Code-Here {framework.ComponentID._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(framework.ComponentID._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(framework.ComponentID._implementation)
    // Insert-Code-Here {framework.ComponentID._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(framework.ComponentID._implementation)

  public:
    // default constructor, used for data wrapping(required)
    ComponentID_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      ComponentID_impl( struct framework_ComponentID__object * ior ) : StubBase(
        ior,true), 
    ::gov::cca::ComponentID((ior==NULL) ? NULL : &((
      *ior).d_gov_cca_componentid)) , _wrapped(false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~ComponentID_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:


    /**
     * Returns the instance name provided in
     * <code>BuilderService.createInstance()</code>
     * or in
     * <code>AbstractFramework.getServices()</code>.
     * @throws CCAException if <code>ComponentID</code> is invalid
     */
    ::std::string
    getInstanceName_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;

    /**
     * Returns a framework specific serialization of the ComponentID.
     * @throws CCAException if <code>ComponentID</code> is
     * invalid.
     */
    ::std::string
    getSerialization_impl() // throws:
    //     ::gov::cca::CCAException
    //     ::sidl::RuntimeException
    ;
  };  // end class ComponentID_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.ComponentID._hmisc)
// insert code here (miscellaneous things)
// DO-NOT-DELETE splicer.end(framework.ComponentID._hmisc)

#endif

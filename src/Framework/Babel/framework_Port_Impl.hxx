// 
// File:          framework_Port_Impl.hxx
// Symbol:        framework.Port-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for framework.Port
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_framework_Port_Impl_hxx
#define included_framework_Port_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_framework_Port_IOR_h
#include "framework_Port_IOR.h"
#endif
#ifndef included_framework_Port_hxx
#include "framework_Port.hxx"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
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


// DO-NOT-DELETE splicer.begin(framework.Port._hincludes)
// insert code here (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(framework.Port._hincludes)

namespace framework { 

  /**
   * Symbol "framework.Port" (version 1.0)
   */
  class Port_impl : public virtual ::framework::Port 
  // DO-NOT-DELETE splicer.begin(framework.Port._inherits)
  // Insert-Code-Here {framework.Port._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(framework.Port._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(framework.Port._implementation)
    // Insert-Code-Here {framework.Port._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(framework.Port._implementation)

  public:
    // default constructor, used for data wrapping(required)
    Port_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      Port_impl( struct framework_Port__object * ior ) : StubBase(ior,true), 
    ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)) , _wrapped(
      false) {_ctor();}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Port_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:

  };  // end class Port_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.Port._hmisc)
// insert code here (miscellaneous things)
// DO-NOT-DELETE splicer.end(framework.Port._hmisc)

#endif

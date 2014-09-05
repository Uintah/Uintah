// 
// File:          framework_Port_Impl.hxx
// Symbol:        framework.Port-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for framework.Port
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_framework_Port_Impl_hxx
#define included_framework_Port_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
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


// DO-NOT-DELETE splicer.begin(framework.Port._includes)
// Insert-Code-Here {framework.Port._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(framework.Port._includes)

namespace framework { 

  /**
   * Symbol "framework.Port" (version 1.0)
   */
  class Port_impl : public virtual UCXX ::framework::Port 
  // DO-NOT-DELETE splicer.begin(framework.Port._inherits)
  // Insert-Code-Here {framework.Port._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(framework.Port._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(framework.Port._implementation)
    // Insert-Code-Here {framework.Port._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(framework.Port._implementation)

  public:
    // default constructor, shouldn't be used (required)
    Port_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      Port_impl( struct framework_Port__object * s ) : StubBase(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~Port_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:

    };  // end class Port_impl

  } // end namespace framework

  // DO-NOT-DELETE splicer.begin(framework.Port._misc)
  // Insert-Code-Here {framework.Port._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(framework.Port._misc)

  #endif

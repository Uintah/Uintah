// 
// File:          who_UIPort_Impl.hxx
// Symbol:        who.UIPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for who.UIPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_who_UIPort_Impl_hxx
#define included_who_UIPort_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
#endif
#ifndef included_who_UIPort_IOR_h
#include "who_UIPort_IOR.h"
#endif
#ifndef included_gov_cca_ports_UIPort_hxx
#include "gov_cca_ports_UIPort.hxx"
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
#ifndef included_who_UIPort_hxx
#include "who_UIPort.hxx"
#endif


// DO-NOT-DELETE splicer.begin(who.UIPort._includes)
// Insert-Code-Here {who.UIPort._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(who.UIPort._includes)

namespace who { 

  /**
   * Symbol "who.UIPort" (version 1.0)
   */
  class UIPort_impl : public virtual UCXX ::who::UIPort 
  // DO-NOT-DELETE splicer.begin(who.UIPort._inherits)
  // Insert-Code-Here {who.UIPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(who.UIPort._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(who.UIPort._implementation)
    // Insert-Code-Here {who.UIPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(who.UIPort._implementation)

  public:
    // default constructor, shouldn't be used (required)
    UIPort_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      UIPort_impl( struct who_UIPort__object * s ) : StubBase(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~UIPort_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:

      /**
       * user defined non-static method.
       */
      int32_t
      ui_impl() ;
    };  // end class UIPort_impl

  } // end namespace who

  // DO-NOT-DELETE splicer.begin(who.UIPort._misc)
  // Insert-Code-Here {who.UIPort._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(who.UIPort._misc)

  #endif

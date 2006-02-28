// 
// File:          who_IDPort_Impl.hxx
// Symbol:        who.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for who.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 

#ifndef included_who_IDPort_Impl_hxx
#define included_who_IDPort_Impl_hxx

#ifndef included_sidl_ucxx_hxx
#include "sidl_ucxx.hxx"
#endif
#ifndef included_who_IDPort_IOR_h
#include "who_IDPort_IOR.h"
#endif
#ifndef included_gov_cca_ports_IDPort_hxx
#include "gov_cca_ports_IDPort.hxx"
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
#ifndef included_who_IDPort_hxx
#include "who_IDPort.hxx"
#endif


// DO-NOT-DELETE splicer.begin(who.IDPort._includes)
// Insert-Code-Here {who.IDPort._includes} (includes or arbitrary code)
// DO-NOT-DELETE splicer.end(who.IDPort._includes)

namespace who { 

  /**
   * Symbol "who.IDPort" (version 1.0)
   */
  class IDPort_impl : public virtual UCXX ::who::IDPort 
  // DO-NOT-DELETE splicer.begin(who.IDPort._inherits)
  // Insert-Code-Here {who.IDPort._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(who.IDPort._inherits)
  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    // DO-NOT-DELETE splicer.begin(who.IDPort._implementation)
    // Insert-Code-Here {who.IDPort._implementation} (additional details)
    // DO-NOT-DELETE splicer.end(who.IDPort._implementation)

  public:
    // default constructor, shouldn't be used (required)
    IDPort_impl() : StubBase(0,true) { } 

      // sidl constructor (required)
      // Note: alternate Skel constructor doesn't call addref()
      // (fixes bug #275)
      IDPort_impl( struct who_IDPort__object * s ) : StubBase(s,
        true) { _ctor(); }

      // user defined construction
      void _ctor();

      // virtual destructor (required)
      virtual ~IDPort_impl() { _dtor(); }

      // user defined destruction
      void _dtor();

      // static class initializer
      static void _load();

    public:


      /**
       * Test prot. Return a string as an ID for Hello component
       */
      ::std::string
      getID_impl() ;
    };  // end class IDPort_impl

  } // end namespace who

  // DO-NOT-DELETE splicer.begin(who.IDPort._misc)
  // Insert-Code-Here {who.IDPort._misc} (miscellaneous things)
  // DO-NOT-DELETE splicer.end(who.IDPort._misc)

  #endif

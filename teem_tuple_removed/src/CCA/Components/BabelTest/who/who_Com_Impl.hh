// 
// File:          who_Com_Impl.hh
// Symbol:        who.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030618 13:12:27 MDT
// Generated:     20030618 13:12:33 MDT
// Description:   Server-side implementation for who.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 13
// source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 

#ifndef included_who_Com_Impl_hh
#define included_who_Com_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_who_Com_IOR_h
#include "who_Com_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_gov_cca_Services_hh
#include "gov_cca_Services.hh"
#endif
#ifndef included_who_Com_hh
#include "who_Com.hh"
#endif


// DO-NOT-DELETE splicer.begin(who.Com._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(who.Com._includes)

namespace who { 

  /**
   * Symbol "who.Com" (version 1.0)
   */
  class Com_impl
  // DO-NOT-DELETE splicer.begin(who.Com._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(who.Com._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Com self;

    // DO-NOT-DELETE splicer.begin(who.Com._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(who.Com._implementation)

  private:
    // private default constructor (required)
    Com_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Com_impl( struct who_Com__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Com_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:


    /**
     * Obtain Services handle, through which the 
     * component communicates with the framework. 
     * This is the one method that every CCA Component
     * must implement. 
     */
    void
    setServices (
      /*in*/ ::gov::cca::Services services
    )
    throw () 
    ;

  };  // end class Com_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.Com._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(who.Com._misc)

#endif

// 
// File:          who_IDPort_Impl.hh
// Symbol:        who.IDPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030618 13:12:24 MDT
// Generated:     20030618 13:12:33 MDT
// Description:   Server-side implementation for who.IDPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/who/who.sidl
// 

#ifndef included_who_IDPort_Impl_hh
#define included_who_IDPort_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_who_IDPort_IOR_h
#include "who_IDPort_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_who_IDPort_hh
#include "who_IDPort.hh"
#endif


// DO-NOT-DELETE splicer.begin(who.IDPort._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(who.IDPort._includes)

namespace who { 

  /**
   * Symbol "who.IDPort" (version 1.0)
   */
  class IDPort_impl
  // DO-NOT-DELETE splicer.begin(who.IDPort._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(who.IDPort._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    IDPort self;

    // DO-NOT-DELETE splicer.begin(who.IDPort._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(who.IDPort._implementation)

  private:
    // private default constructor (required)
    IDPort_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    IDPort_impl( struct who_IDPort__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~IDPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:


    /**
     * Test prot. Return a string as an ID for Hello component
     */
    ::std::string
    getID() throw () 
    ;
  };  // end class IDPort_impl

} // end namespace who

// DO-NOT-DELETE splicer.begin(who.IDPort._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(who.IDPort._misc)

#endif

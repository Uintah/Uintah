// 
// File:          NewPort_StringPort_Impl.hh
// Symbol:        NewPort.StringPort-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20040301 18:38:02 MST
// Generated:     20040301 18:38:04 MST
// Description:   Server-side implementation for NewPort.StringPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 15
// source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/NewPort/NewPort.sidl
// 

#ifndef included_NewPort_StringPort_Impl_hh
#define included_NewPort_StringPort_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_NewPort_StringPort_IOR_h
#include "NewPort_StringPort_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_NewPort_StringPort_hh
#include "NewPort_StringPort.hh"
#endif
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif


// DO-NOT-DELETE splicer.begin(NewPort.StringPort._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(NewPort.StringPort._includes)

namespace NewPort { 

  /**
   * Symbol "NewPort.StringPort" (version 1.0)
   */
  class StringPort_impl
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    StringPort self;

    // DO-NOT-DELETE splicer.begin(NewPort.StringPort._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(NewPort.StringPort._implementation)

  private:
    // private default constructor (required)
    StringPort_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    StringPort_impl( struct NewPort_StringPort__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~StringPort_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

    /**
     * user defined non-static method.
     */
    ::std::string
    getString() throw () 
    ;
  };  // end class StringPort_impl

} // end namespace NewPort

// DO-NOT-DELETE splicer.begin(NewPort.StringPort._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(NewPort.StringPort._misc)

#endif

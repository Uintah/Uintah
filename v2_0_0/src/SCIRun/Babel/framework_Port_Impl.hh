// 
// File:          framework_Port_Impl.hh
// Symbol:        framework.Port-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030720 10:32:36 MDT
// Generated:     20030720 10:32:38 MDT
// Description:   Server-side implementation for framework.Port
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 14
// source-url    = file:/home/sci/kzhang/SCIRun/debug/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_Port_Impl_hh
#define included_framework_Port_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_Port_IOR_h
#include "framework_Port_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_framework_Port_hh
#include "framework_Port.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.Port._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.Port._includes)

namespace framework { 

  /**
   * Symbol "framework.Port" (version 1.0)
   */
  class Port_impl
  // DO-NOT-DELETE splicer.begin(framework.Port._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.Port._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Port self;

    // DO-NOT-DELETE splicer.begin(framework.Port._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(framework.Port._implementation)

  private:
    // private default constructor (required)
    Port_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Port_impl( struct framework_Port__object * s ) : self(s,true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Port_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

  };  // end class Port_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.Port._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.Port._misc)

#endif

// 
// File:          framework_ComponentID_Impl.hh
// Symbol:        framework.ComponentID-v1.0
// Symbol Type:   class
// Babel Version: 0.7.0
// SIDL Created:  20020730 13:51:30 MST
// Generated:     20020730 13:51:36 MST
// Description:   Server-side implementation for framework.ComponentID
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_framework_ComponentID_Impl_hh
#define included_framework_ComponentID_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_ComponentID_IOR_h
#include "framework_ComponentID_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_framework_ComponentID_hh
#include "framework_ComponentID.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.ComponentID._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.ComponentID._includes)

namespace framework { 

  /**
   * Symbol "framework.ComponentID" (version 1.0)
   */
  class ComponentID_impl
  // DO-NOT-DELETE splicer.begin(framework.ComponentID._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.ComponentID._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    ComponentID self;

    // DO-NOT-DELETE splicer.begin(framework.ComponentID._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(framework.ComponentID._implementation)

  private:
    // private default constructor (required)
    ComponentID_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    ComponentID_impl( struct framework_ComponentID__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~ComponentID_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:


    /**
     * Produce a string that, within the current framework, uniquely defines 
     * this component reference. 
     */
    std::string
    toString() throw () 
    ;
  };  // end class ComponentID_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.ComponentID._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.ComponentID._misc)

#endif

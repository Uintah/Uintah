// 
// File:          framework_TypeMap_Impl.hh
// Symbol:        framework.TypeMap-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20021108 00:42:48 EST
// Generated:     20021108 00:42:50 EST
// Description:   Server-side implementation for framework.TypeMap
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 17
// source-url    = file:/.automount/linbox1/root/home/user2/sparker/SCIRun/cca/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_TypeMap_Impl_hh
#define included_framework_TypeMap_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_TypeMap_IOR_h
#include "framework_TypeMap_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_framework_TypeMap_hh
#include "framework_TypeMap.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.TypeMap._includes)
// Put additional includes or other arbitrary code here...
// DO-NOT-DELETE splicer.end(framework.TypeMap._includes)

namespace framework { 

  /**
   * Symbol "framework.TypeMap" (version 1.0)
   */
  class TypeMap_impl
  // DO-NOT-DELETE splicer.begin(framework.TypeMap._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.TypeMap._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    TypeMap self;

    // DO-NOT-DELETE splicer.begin(framework.TypeMap._implementation)
    // Put additional implementation details here...
    // DO-NOT-DELETE splicer.end(framework.TypeMap._implementation)

  private:
    // private default constructor (required)
    TypeMap_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    TypeMap_impl( struct framework_TypeMap__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~TypeMap_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

    /**
     * user defined non-static method.
     */
    void
    temp() throw () 
    ;
  };  // end class TypeMap_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.TypeMap._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.TypeMap._misc)

#endif

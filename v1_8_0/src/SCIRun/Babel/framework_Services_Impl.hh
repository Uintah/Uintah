// 
// File:          framework_Services_Impl.hh
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20021109 17:19:36 MST
// Generated:     20021109 17:19:39 MST
// Description:   Server-side implementation for framework.Services
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sparker/SCIRun/cca/../src/SCIRun/Babel/framework.sidl
// 

#ifndef included_framework_Services_Impl_hh
#define included_framework_Services_Impl_hh

#ifndef included_SIDL_cxx_hh
#include "SIDL_cxx.hh"
#endif
#ifndef included_framework_Services_IOR_h
#include "framework_Services_IOR.h"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_SIDL_BaseInterface_hh
#include "SIDL_BaseInterface.hh"
#endif
#ifndef included_framework_Services_hh
#include "framework_Services.hh"
#endif
#ifndef included_govcca_ComponentID_hh
#include "govcca_ComponentID.hh"
#endif
#ifndef included_govcca_Port_hh
#include "govcca_Port.hh"
#endif
#ifndef included_govcca_TypeMap_hh
#include "govcca_TypeMap.hh"
#endif


// DO-NOT-DELETE splicer.begin(framework.Services._includes)
// Put additional includes or other arbitrary code here...
#include <SCIRun/Babel/BabelPortInstance.h>
using namespace SCIRun;
// DO-NOT-DELETE splicer.end(framework.Services._includes)

namespace framework { 

  /**
   * Symbol "framework.Services" (version 1.0)
   */
  class Services_impl
  // DO-NOT-DELETE splicer.begin(framework.Services._inherits)
  // Put additional inheritance here...
  // DO-NOT-DELETE splicer.end(framework.Services._inherits)
  {

  private:
    // Pointer back to IOR.
    // Use this to dispatch back through IOR vtable.
    Services self;

    // DO-NOT-DELETE splicer.begin(framework.Services._implementation)
    std::map<std::string, PortInstance*> ports;
    govcca::Component component;
    // DO-NOT-DELETE splicer.end(framework.Services._implementation)

  private:
    // private default constructor (required)
    Services_impl() {} 

  public:
    // SIDL constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
    Services_impl( struct framework_Services__object * s ) : self(s,
      true) { _ctor(); }

    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~Services_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

  public:

    /**
     * user defined non-static method.
     */
    void*
    getData() throw () 
    ;

    /**
     * Ask for a previously registered Port; will return a Port or generate an error. 
     */
    ::govcca::Port
    getPort (
      /*in*/ const ::std::string& name
    )
    throw () 
    ;


    /**
     * Ask for a previously registered Port and return that Port if it is
     * available or return null otherwise. 
     */
    ::govcca::Port
    getPortNonblocking (
      /*in*/ const ::std::string& name
    )
    throw () 
    ;


    /**
     * Modified according to Motion 31 
     */
    void
    registerUsesPort (
      /*in*/ const ::std::string& name,
      /*in*/ const ::std::string& type,
      /*in*/ ::govcca::TypeMap properties
    )
    throw () 
    ;


    /**
     * Notify the framework that a Port, previously registered by this component,
     * is no longer needed. 
     */
    void
    unregisterUsesPort (
      /*in*/ const ::std::string& name
    )
    throw () 
    ;


    /**
     * Exports a Port implemented by this component to the framework.  
     * This Port is now available for the framework to connect to other components. 
     * Modified according to Motion 31 
     */
    void
    addProvidesPort (
      /*in*/ ::govcca::Port inPort,
      /*in*/ const ::std::string& name,
      /*in*/ const ::std::string& type,
      /*in*/ ::govcca::TypeMap properties
    )
    throw () 
    ;


    /**
     * Notifies the framework that a previously exported Port is no longer 
     * available for use.
     */
    void
    removeProvidesPort (
      /*in*/ const ::std::string& name
    )
    throw () 
    ;


    /**
     * Notifies the framework that this component is finished with this Port.   
     * releasePort() method calls exactly match getPort() mehtod calls.  After 
     * releasePort() is invoked all references to the released Port become invalid. 
     */
    void
    releasePort (
      /*in*/ const ::std::string& name
    )
    throw () 
    ;

    /**
     * user defined non-static method.
     */
    ::govcca::TypeMap
    createTypeMap() throw () 
    ;
    /**
     * user defined non-static method.
     */
    ::govcca::TypeMap
    getPortProperties (
      /*in*/ const ::std::string& portName
    )
    throw () 
    ;


    /**
     * Get a reference to the component to which this Services object belongs. 
     */
    ::govcca::ComponentID
    getComponentID() throw () 
    ;
  };  // end class Services_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.Services._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.Services._misc)

#endif

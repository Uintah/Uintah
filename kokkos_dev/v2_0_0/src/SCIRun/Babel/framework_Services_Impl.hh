// 
// File:          framework_Services_Impl.hh
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030720 10:32:33 MDT
// Generated:     20030720 10:32:38 MDT
// Description:   Server-side implementation for framework.Services
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/home/sci/kzhang/SCIRun/debug/../src/SCIRun/Babel/framework.sidl
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
#ifndef included_gov_cca_CCAException_hh
#include "gov_cca_CCAException.hh"
#endif
#ifndef included_gov_cca_ComponentID_hh
#include "gov_cca_ComponentID.hh"
#endif
#ifndef included_gov_cca_Port_hh
#include "gov_cca_Port.hh"
#endif
#ifndef included_gov_cca_TypeMap_hh
#include "gov_cca_TypeMap.hh"
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
    gov::cca::Component component;
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
     * Fetch a previously registered Port (defined by either 
     * addProvidePort or (more typically) registerUsesPort).  
     * @return Will return the Port (possibly waiting forever while
     * attempting to acquire it) or throw an exception. Does not return
     * NULL, even in the case where no connection has been made. 
     * If a Port is returned,
     * there is then a contract that the port will remain valid for use
     * by the caller until the port is released via releasePort(), or a 
     * Disconnect Event is successfully dispatched to the caller,
     * or a runtime exception (such as network failure) occurs during 
     * invocation of some function in the Port. 
     * &lt;p&gt;
     * Subtle interpretation: If the Component is not listening for
     * Disconnect events, then the framework has no clean way to
     * break the connection until after the component calls releasePort.
     * &lt;/p&gt;
     * &lt;p&gt;The framework may go through some machinations to obtain
     *    the port, possibly involving an interactive user or network 
     *    queries, before giving up and throwing an exception.
     * &lt;/p&gt;
     * 
     * @param portName The previously registered or provide port which
     * 	   the component now wants to use.
     * @exception CCAException with the following types: NotConnected, PortNotDefined, 
     *                NetworkError, OutOfMemory.
     */
    ::gov::cca::Port
    getPort (
      /*in*/ const ::std::string& portName
    )
    throw ( 
      ::gov::cca::CCAException
    );


    /**
     * Get a previously registered Port (defined by
     * either addProvide or registerUses) and return that
     * Port if it is available immediately (already connected
     * without further connection machinations).
     * There is an contract that the
     * port will remain valid per the description of getPort.
     * @return The named port, if it exists and is connected or self-provided,
     * 	      or NULL if it is registered and is not yet connected. Does not
     * 	      return if the Port is neither registered nor provided, but rather
     * 	      throws an exception.
     * @param portName registered or provided port that
     * 	     the component now wants to use.
     * @exception CCAException with the following types: PortNotDefined, OutOfMemory.
     */
    ::gov::cca::Port
    getPortNonblocking (
      /*in*/ const ::std::string& portName
    )
    throw ( 
      ::gov::cca::CCAException
    );


    /**
     * Notifies the framework that this component is finished 
     * using the previously fetched Port that is named.     
     * The releasePort() method calls should be paired with 
     * getPort() method calls; however, an extra call to releasePort()
     * for the same name may (is not required to) generate an exception.
     * Calls to release ports which are not defined or have never be fetched
     * with one of the getPort functions generate exceptions.
     * @param portName The name of a port.
     * @exception CCAException with the following types: PortNotDefined, PortNotInUse.
     */
    void
    releasePort (
      /*in*/ const ::std::string& portName
    )
    throw ( 
      ::gov::cca::CCAException
    );


    /**
     * Creates a TypeMap, potentially to be used in subsequent
     * calls to describe a Port.  Initially, this map is empty.
     */
    ::gov::cca::TypeMap
    createTypeMap() throw ( 
      ::gov::cca::CCAException
    );

    /**
     * Register a request for a Port that will be retrieved subsequently 
     * with a call to getPort().
     * @param portName A string uniquely describing this port.  This string
     * must be unique for this component, over both uses and provides ports.
     * @param type A string desribing the type of this port.
     * @param properties A TypeMap describing optional properties
     * associated with this port. This can be a null pointer, which
     * indicates an empty list of properties.  Properties may be
     * obtained from createTypeMap or any other source.  The properties
     * be copied into the framework, and subsequent changes to the
     * properties object will have no effect on the properties
     * associated with this port.
     * In these properties, all frameworks recognize at least the
     * following keys and values in implementing registerUsesPort:
     * &lt;pre&gt;
     * key:              standard values (in string form)     default
     * &quot;MAX_CONNECTIONS&quot; any nonnegative integer, &quot;unlimited&quot;.   1
     * &quot;MIN_CONNECTIONS&quot; any integer &gt; 0.                        0
     * &quot;ABLE_TO_PROXY&quot;   &quot;true&quot;, &quot;false&quot;                      &quot;false&quot;
     * &lt;/pre&gt;
     * The component is not expected to work if the framework
     * has not satisfied the connection requirements.
     * The framework is allowed to return an error if it
     * is incapable of meeting the connection requirements,
     * e.g. it does not implement multiple uses ports.
     * The caller of registerUsesPort is not obligated to define
     * these properties. If left undefined, the default listed above is
     *       assumed.
     * @exception CCAException with the following types: PortAlreadyDefined, OutOfMemory.
     */
    void
    registerUsesPort (
      /*in*/ const ::std::string& portName,
      /*in*/ const ::std::string& type,
      /*in*/ ::gov::cca::TypeMap properties
    )
    throw ( 
      ::gov::cca::CCAException
    );


    /**
     * Notify the framework that a Port, previously registered by this
     * component but currently not in use, is no longer desired. 
     * Unregistering a port that is currently 
     * in use (i.e. an unreleased getPort() being outstanding) 
     * is an error.
     * @param name The name of a registered Port.
     * @exception CCAException with the following types: UsesPortNotReleased, PortNotDefined.
     */
    void
    unregisterUsesPort (
      /*in*/ const ::std::string& portName
    )
    throw ( 
      ::gov::cca::CCAException
    );


    /**
     * Exposes a Port from this component to the framework.  
     * This Port is now available for the framework to connect 
     * to other components. 
     * @param inPort An abstract interface (tagged with CCA-ness
     * 	by inheriting from gov.cca.Port) the framework will
     * 	make available to other components.
     * 
     * @param portName string uniquely describing this port.  This string
     * must be unique for this component, over both uses and provides ports.
     * 
     * @param type string describing the type (class) of this port.
     * 
     * @param properties A TypeMap describing optional properties
     * associated with this port. This can be a null pointer, which
     * indicates an empty list of properties.  Properties may be
     * obtained from createTypeMap or any other source.  The properties
     * be copied into the framework, and subsequent changes to the
     * properties object will have no effect on the properties
     * associated with this port.
     * In these properties, all frameworks recognize at least the
     * following keys and values in implementing registerUsesPort:
     * &lt;pre&gt;
     * key:              standard values (in string form)     default
     * &quot;MAX_CONNECTIONS&quot; any nonnegative integer, &quot;unlimited&quot;.   1
     * &quot;MIN_CONNECTIONS&quot; any integer &gt; 0.                        0
     * &quot;ABLE_TO_PROXY&quot;   &quot;true&quot;, &quot;false&quot;                      &quot;false&quot;
     * &lt;/pre&gt;
     * The component is not expected to work if the framework
     * has not satisfied the connection requirements.
     * The framework is allowed to return an error if it
     * is incapable of meeting the connection requirements,
     * e.g. it does not implement multiple uses ports.
     * The caller of addProvidesPort is not obligated to define
     * these properties. If left undefined, the default listed above is
     * assumed.
     * @exception CCAException with the following types: PortAlreadyDefined, OutOfMemory.
     */
    void
    addProvidesPort (
      /*in*/ ::gov::cca::Port inPort,
      /*in*/ const ::std::string& portName,
      /*in*/ const ::std::string& type,
      /*in*/ ::gov::cca::TypeMap properties
    )
    throw ( 
      ::gov::cca::CCAException
    );


    /**
     * Returns the complete list of the properties for a Port.  This
     * includes the properties defined when the port was registered
     * (these properties can be modified by the framework), two special
     * properties &quot;cca.portName&quot; and &quot;cca.portType&quot;, and any other
     * properties that the framework wishes to disclose to the component.
     * The framework may also choose to provide only the subset of input
     * properties (i.e. from addProvidesPort/registerUsesPort) that it
     * will honor.      
     */
    ::gov::cca::TypeMap
    getPortProperties (
      /*in*/ const ::std::string& name
    )
    throw () 
    ;


    /**
     * Notifies the framework that a previously exposed Port is no longer 
     * available for use. The Port being removed must exist
     * until this call returns, or a CCAException may occur.
     * @param name The name of a provided Port.
     * @exception PortNotDefined. In general, the framework will not dictate 
     * when the component chooses to stop offering services.
     */
    void
    removeProvidesPort (
      /*in*/ const ::std::string& portName
    )
    throw ( 
      ::gov::cca::CCAException
    );


    /**
     * Get a reference to the component to which this 
     * Services object belongs. 
     */
    ::gov::cca::ComponentID
    getComponentID() throw () 
    ;
  };  // end class Services_impl

} // end namespace framework

// DO-NOT-DELETE splicer.begin(framework.Services._misc)
// Put miscellaneous things here...
// DO-NOT-DELETE splicer.end(framework.Services._misc)

#endif

// 
// File:          framework_Services_Impl.cxx
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.11.0
// Description:   Server-side implementation for framework.Services
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.11.0
// 
#include "framework_Services_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_ComponentRelease_hxx
#include "gov_cca_ComponentRelease.hxx"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(framework.Services._includes)
// Insert-Code-Here {framework.Services._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(framework.Services._includes)

// user defined constructor
void framework::Services_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.Services._ctor)
  // Insert-Code-Here {framework.Services._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(framework.Services._ctor)
}

// user defined destructor
void framework::Services_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.Services._dtor)
  // Insert-Code-Here {framework.Services._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(framework.Services._dtor)
}

// static class initializer
void framework::Services_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.Services._load)
  // Insert-Code-Here {framework.Services._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.Services._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  getData[]
 */
void*
framework::Services_impl::getData_impl () 

{
  // DO-NOT-DELETE splicer.begin(framework.Services.getData)
  // Insert-Code-Here {framework.Services.getData} (getData method)
  // DO-NOT-DELETE splicer.end(framework.Services.getData)
}

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
 * <p>
 * Subtle interpretation: If the Component is not listening for
 * Disconnect events, then the framework has no clean way to
 * break the connection until after the component calls releasePort.
 * </p>
 * <p>The framework may go through some machinations to obtain
 *    the port, possibly involving an interactive user or network
 *    queries, before giving up and throwing an exception.
 * </p>
 * 
 * @param portName The previously registered or provide port which
 *         the component now wants to use.
 * @exception CCAException with the following types: NotConnected, PortNotDefined,
 *                NetworkError, OutOfMemory.
 */
UCXX ::gov::cca::Port
framework::Services_impl::getPort_impl (
  /* in */const ::std::string& portName ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.getPort)
  // Insert-Code-Here {framework.Services.getPort} (getPort method)
  // DO-NOT-DELETE splicer.end(framework.Services.getPort)
}

/**
 * Get a previously registered Port (defined by
 * either addProvide or registerUses) and return that
 * Port if it is available immediately (already connected
 * without further connection machinations).
 * There is an contract that the
 * port will remain valid per the description of getPort.
 * @return The named port, if it exists and is connected or self-provided,
 *            or NULL if it is registered and is not yet connected. Does not
 *            return if the Port is neither registered nor provided, but rather
 *            throws an exception.
 * @param portName registered or provided port that
 *           the component now wants to use.
 * @exception CCAException with the following types: PortNotDefined, OutOfMemory.
 */
UCXX ::gov::cca::Port
framework::Services_impl::getPortNonblocking_impl (
  /* in */const ::std::string& portName ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.getPortNonblocking)
  // Insert-Code-Here {framework.Services.getPortNonblocking} (getPortNonblocking method)
  // DO-NOT-DELETE splicer.end(framework.Services.getPortNonblocking)
}

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
framework::Services_impl::releasePort_impl (
  /* in */const ::std::string& portName ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.releasePort)
  // Insert-Code-Here {framework.Services.releasePort} (releasePort method)
  // DO-NOT-DELETE splicer.end(framework.Services.releasePort)
}

/**
 * Creates a TypeMap, potentially to be used in subsequent
 * calls to describe a Port.  Initially, this map is empty.
 */
UCXX ::gov::cca::TypeMap
framework::Services_impl::createTypeMap_impl () 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
)
{
  // DO-NOT-DELETE splicer.begin(framework.Services.createTypeMap)
  // Insert-Code-Here {framework.Services.createTypeMap} (createTypeMap method)
  // DO-NOT-DELETE splicer.end(framework.Services.createTypeMap)
}

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
 * <pre xml:space="preserve">
 * key:              standard values (in string form)     default
 * "MAX_CONNECTIONS" any nonnegative integer, "unlimited".   1
 * "MIN_CONNECTIONS" any integer > 0.                        0
 * "ABLE_TO_PROXY"   "true", "false"                      "false"
 * </pre>
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
framework::Services_impl::registerUsesPort_impl (
  /* in */const ::std::string& portName,
  /* in */const ::std::string& type,
  /* in */UCXX ::gov::cca::TypeMap properties ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.registerUsesPort)
  // Insert-Code-Here {framework.Services.registerUsesPort} (registerUsesPort method)
  // DO-NOT-DELETE splicer.end(framework.Services.registerUsesPort)
}

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
framework::Services_impl::unregisterUsesPort_impl (
  /* in */const ::std::string& portName ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.unregisterUsesPort)
  // Insert-Code-Here {framework.Services.unregisterUsesPort} (unregisterUsesPort method)
  // DO-NOT-DELETE splicer.end(framework.Services.unregisterUsesPort)
}

/**
 * Exposes a Port from this component to the framework.
 * This Port is now available for the framework to connect
 * to other components.
 * @param inPort An abstract interface (tagged with CCA-ness
 *      by inheriting from gov.cca.Port) the framework will
 *      make available to other components.
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
 * <pre xml:space="preserve">
 * key:              standard values (in string form)     default
 * "MAX_CONNECTIONS" any nonnegative integer, "unlimited".   1
 * "MIN_CONNECTIONS" any integer > 0.                        0
 * "ABLE_TO_PROXY"   "true", "false"                      "false"
 * </pre>
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
framework::Services_impl::addProvidesPort_impl (
  /* in */UCXX ::gov::cca::Port inPort,
  /* in */const ::std::string& portName,
  /* in */const ::std::string& type,
  /* in */UCXX ::gov::cca::TypeMap properties ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.addProvidesPort)
  // Insert-Code-Here {framework.Services.addProvidesPort} (addProvidesPort method)
  // DO-NOT-DELETE splicer.end(framework.Services.addProvidesPort)
}

/**
 * Returns the complete list of the properties for a Port.  This
 * includes the properties defined when the port was registered
 * (these properties can be modified by the framework), two special
 * properties "cca.portName" and "cca.portType", and any other
 * properties that the framework wishes to disclose to the component.
 * The framework may also choose to provide only the subset of input
 * properties (i.e. from addProvidesPort/registerUsesPort) that it
 * will honor.
 */
UCXX ::gov::cca::TypeMap
framework::Services_impl::getPortProperties_impl (
  /* in */const ::std::string& name ) 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.getPortProperties)
  // Insert-Code-Here {framework.Services.getPortProperties} (getPortProperties method)
  // DO-NOT-DELETE splicer.end(framework.Services.getPortProperties)
}

/**
 * Notifies the framework that a previously exposed Port is no longer
 * available for use. The Port being removed must exist
 * until this call returns, or a CCAException may occur.
 * @param name The name of a provided Port.
 * @exception PortNotDefined. In general, the framework will not dictate
 * when the component chooses to stop offering services.
 */
void
framework::Services_impl::removeProvidesPort_impl (
  /* in */const ::std::string& portName ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.removeProvidesPort)
  // Insert-Code-Here {framework.Services.removeProvidesPort} (removeProvidesPort method)
  // DO-NOT-DELETE splicer.end(framework.Services.removeProvidesPort)
}

/**
 * Get a reference to the component to which this
 * Services object belongs.
 */
UCXX ::gov::cca::ComponentID
framework::Services_impl::getComponentID_impl () 

{
  // DO-NOT-DELETE splicer.begin(framework.Services.getComponentID)
  // Insert-Code-Here {framework.Services.getComponentID} (getComponentID method)
  // DO-NOT-DELETE splicer.end(framework.Services.getComponentID)
}

/**
 * Obtain a callback for component destruction.
 * @param callback an object that implements the ComponentRelease
 * interface that will be called when the component is to be destroyed.
 * 
 * Register a callback to be executed when the component is going
 * to be destroyed.  During this callback, the Services object passed
 * through setServices will still be valid, but after all such
 * callbacks are made for a specific component, subsequent usage
 * of the Services object is not allowed/is undefined.
 */
void
framework::Services_impl::registerForRelease_impl (
  /* in */UCXX ::gov::cca::ComponentRelease callback ) 
throw ( 
  UCXX ::gov::cca::CCAException, 
  UCXX ::sidl::RuntimeException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.registerForRelease)
  // Insert-Code-Here {framework.Services.registerForRelease} (registerForRelease method)
  // DO-NOT-DELETE splicer.end(framework.Services.registerForRelease)
}


// DO-NOT-DELETE splicer.begin(framework.Services._misc)
// Insert-Code-Here {framework.Services._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(framework.Services._misc)


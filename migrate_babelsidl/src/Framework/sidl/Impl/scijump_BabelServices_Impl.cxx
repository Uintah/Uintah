// 
// File:          scijump_BabelServices_Impl.cxx
// Symbol:        scijump.BabelServices-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.BabelServices
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_BabelServices_Impl.hxx"

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
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_scijump_SCIJumpFramework_hxx
#include "scijump_SCIJumpFramework.hxx"
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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(scijump.BabelServices._includes)

#include "scijump.hxx"

#include <iostream>

using namespace SCIRun;

// Insert-Code-Here {scijump.BabelServices._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.BabelServices._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::BabelServices_impl::BabelServices_impl() : StubBase(reinterpret_cast< 
  void*>(::scijump::BabelServices::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices._ctor2)
  // Insert-Code-Here {scijump.BabelServices._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.BabelServices._ctor2)
}

// user defined constructor
void scijump::BabelServices_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices._ctor)

  lock_ports = new SCIRun::Mutex("BabelServices::ports lock");
  lock_services = new SCIRun::Mutex("BabelServices::services lock");

  // DO-NOT-DELETE splicer.end(scijump.BabelServices._ctor)
}

// user defined destructor
void scijump::BabelServices_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices._dtor)

  delete lock_ports;
  delete lock_services;

  // DO-NOT-DELETE splicer.end(scijump.BabelServices._dtor)
}

// static class initializer
void scijump::BabelServices_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices._load)
  // Insert-Code-Here {scijump.BabelServices._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.BabelServices._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::BabelServices_impl::initialize_impl (
  /* in */::scijump::SCIJumpFramework& framework,
  /* in */const ::std::string& selfInstanceName,
  /* in */const ::std::string& selfClassName,
  /* in */::gov::cca::TypeMap& selfProperties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.initialize)
  this->framework = framework;
  this->selfInstanceName = selfInstanceName;
  this->selfClassName = selfClassName;
  if (selfProperties._is_nil()) {
    this->selfProperties = scijump::TypeMap::_create();
  } else {
    this->selfProperties = selfProperties;
  }
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.initialize)
}

/**
 * Method:  getPortInfo[]
 */
::sci::cca::core::PortInfo
scijump::BabelServices_impl::getPortInfo_impl (
  /* in */const ::std::string& name ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.getPortInfo)
  PortMap::iterator iter = ports.find(name);
  if ( iter == ports.end() ) {
    return NULL;
  }
  return iter->second;
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.getPortInfo)
}

/**
 * Method:  getPorts[]
 */
::sidl::array< ::sci::cca::core::PortInfo>
scijump::BabelServices_impl::getPorts_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.getPorts)
  // Insert-Code-Here {scijump.BabelServices.getPorts} (getPorts method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelServices.getPorts)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPorts");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelServices.getPorts)
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.getPorts)
}

/**
 *  
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
 * the port, possibly involving an interactive user or network 
 * queries, before giving up and throwing an exception.
 * </p>
 * 
 * @param portName The previously registered or provide port which
 * the component now wants to use.
 * @exception CCAException with the following types: NotConnected, PortNotDefined, 
 * NetworkError, OutOfMemory.
 */
::gov::cca::Port
scijump::BabelServices_impl::getPort_impl (
  /* in */const ::std::string& portName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.getPort)

  Guard guard(lock_ports);

  PortMap::iterator iter = ports.find(portName);
  if ( iter == ports.end() ) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_PortNotDefined);
    ex.setNote("Port [" + portName + "] does not exist");
    ex.add(__FILE__, __LINE__, "getPort");
    throw ex;
  }
  ::sci::cca::core::PortInfo pi = iter->second;
  if (pi.getPortType() == ::sci::cca::core::PortType_ProvidesPort) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_BadPortName);
    ex.setNote("cannot call getPort on a provides port");
    ex.add(__FILE__, __LINE__, "getPort");
    throw ex;
  }

  // scijump framework connects framework services to uses ports
  if (! pi.isConnected() ) {
    if ( ! framework.isFrameworkService( pi.getClass() ) ) {
      scijump::CCAException ex = scijump::CCAException::_create();
      ex.initialize(::gov::cca::CCAExceptionType_PortNotConnected);
      ex.setNote("Port [" + portName + "] is not connected");
      ex.add(__FILE__, __LINE__, "getPort");
      throw ex;
    }
    // (from Plume) ask for the service: the framework will also make the connection
    ::sci::cca::core::ServiceInfo service = framework.getFrameworkService(pi.getClass(), pi);
    Guard guard(lock_services);
    servicePorts[portName] = service;
  }

  // port is connected
  pi.incrementUseCount();
  return pi.getPeer().getPort();

  // DO-NOT-DELETE splicer.end(scijump.BabelServices.getPort)
}

/**
 *  
 * Get a previously registered Port (defined by
 * either addProvide or registerUses) and return that
 * Port if it is available immediately (already connected
 * without further connection machinations).
 * There is an contract that the
 * port will remain valid per the description of getPort.
 * @return The named port, if it exists and is connected or self-provided,
 * or NULL if it is registered and is not yet connected. Does not
 * return if the Port is neither registered nor provided, but rather
 * throws an exception.
 * @param portName registered or provided port that
 * the component now wants to use.
 * @exception CCAException with the following types: PortNotDefined, OutOfMemory.
 */
::gov::cca::Port
scijump::BabelServices_impl::getPortNonblocking_impl (
  /* in */const ::std::string& portName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.getPortNonblocking)
  // Insert-Code-Here {scijump.BabelServices.getPortNonblocking} (getPortNonblocking method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelServices.getPortNonblocking)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPortNonblocking");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelServices.getPortNonblocking)
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.getPortNonblocking)
}

/**
 *  
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
scijump::BabelServices_impl::releasePort_impl (
  /* in */const ::std::string& portName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.releasePort)
  ::sci::cca::core::PortInfo pi;
  {
    Guard guard(lock_ports);
    PortMap::iterator iter = ports.find(portName);
    if ( iter == ports.end() ) {
      scijump::CCAException ex = scijump::CCAException::_create();
      ex.initialize(::gov::cca::CCAExceptionType_PortNotDefined);
      ex.setNote("Port [" + portName + "] does not exist");
      ex.add(__FILE__, __LINE__, "releasePort");
      throw ex;
    }
    pi = iter->second;

    if ( pi.getPortType() == ::sci::cca::core::PortType_ProvidesPort ) {
      scijump::CCAException ex = scijump::CCAException::_create();
      ex.initialize(::gov::cca::CCAExceptionType_PortNotDefined);
      ex.setNote("Cannot release a provides port");
      ex.add(__FILE__, __LINE__, "releasePort");
      throw ex;
    }

    if ( ! pi.decrementUseCount()) {
      scijump::CCAException ex = scijump::CCAException::_create();
      ex.initialize(::gov::cca::CCAExceptionType_PortNotInUse);
      ex.setNote("Port [" + portName + "] released without corresponding get");
      ex.add(__FILE__, __LINE__, "releasePort");
      throw ex;
    }
  }
  // release the lock_ports as we may need it if we
  // also release a framework service
  {
    Guard guard(lock_services);

    if ( ! pi.inUse() ) {
      ServicePortMap::iterator iter = servicePorts.find(portName);
      if ( iter != servicePorts.end() ) {
        ::sci::cca::core::ServiceInfo si = iter->second;
        servicePorts.erase(iter);
        framework.releaseFrameworkService(si);
      }
    }
  }
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.releasePort)
}

/**
 * Creates a TypeMap, potentially to be used in subsequent
 * calls to describe a Port.  Initially, this map is empty.
 */
::gov::cca::TypeMap
scijump::BabelServices_impl::createTypeMap_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.createTypeMap)
  return scijump::TypeMap::_create();
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.createTypeMap)
}

/**
 *  
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
 * assumed.
 * @exception CCAException with the following types: PortAlreadyDefined, OutOfMemory.
 */
void
scijump::BabelServices_impl::registerUsesPort_impl (
  /* in */const ::std::string& portName,
  /* in */const ::std::string& type,
  /* in */::gov::cca::TypeMap& properties ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.registerUsesPort)

  Guard guard(lock_ports);

  if ( ports.find(portName) != ports.end() ) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_PortAlreadyDefined);
    ex.setNote("Port [" + portName + "] already exists");
    ex.add(__FILE__, __LINE__, "registerUsesPort");
    throw ex;
  }

  scijump::BabelPortInfo pi = scijump::BabelPortInfo::_create();
  pi.initialize(portName, type, ::sci::cca::core::PortType_UsesPort, properties);
  ports[portName] = pi;

  std::cerr << "scijump::Services_impl::registerUsesPort_impl: " << portName << std::endl;

  // DO-NOT-DELETE splicer.end(scijump.BabelServices.registerUsesPort)
}

/**
 *  
 * Notify the framework that a Port, previously registered by this
 * component but currently not in use, is no longer desired. 
 * Unregistering a port that is currently 
 * in use (i.e. an unreleased getPort() being outstanding) 
 * is an error.
 * @param portName The name of a registered Port.
 * @exception CCAException with the following types: UsesPortNotReleased, PortNotDefined.
 */
void
scijump::BabelServices_impl::unregisterUsesPort_impl (
  /* in */const ::std::string& portName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.unregisterUsesPort)

  Guard guard(lock_ports);

  PortMap::iterator iter = ports.find(portName);
  if ( iter == ports.end() ) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_PortNotDefined);
    ex.setNote("Port [" + portName + "] does not exist");
    ex.add(__FILE__, __LINE__, "unregisterUsesPort");
    throw ex;
  }

  ::sci::cca::core::PortInfo pi = iter->second;
  if (pi.isConnected()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_UsesPortNotReleased);
    ex.setNote("Can not release port [" + portName + "]: port in use");
    ex.add(__FILE__, __LINE__, "unregisterUsesPort");
    throw ex;
  }

  ports.erase(iter);
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.unregisterUsesPort)
}

/**
 *  
 * Exposes a Port from this component to the framework.  
 * This Port is now available for the framework to connect 
 * to other components. 
 * @param inPort An abstract interface (tagged with CCA-ness
 * by inheriting from gov.cca.Port) the framework will
 * make available to other components.
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
scijump::BabelServices_impl::addProvidesPort_impl (
  /* in */::gov::cca::Port& inPort,
  /* in */const ::std::string& portName,
  /* in */const ::std::string& type,
  /* in */::gov::cca::TypeMap& properties ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.addProvidesPort)

  Guard guard(lock_ports);

  if ( ports.find(portName) != ports.end() ) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_PortAlreadyDefined);
    ex.setNote("Port [" + portName + "] already exists");
    ex.add(__FILE__, __LINE__, "registerUsesPort");
    throw ex;
  }

  scijump::BabelPortInfo pi = scijump::BabelPortInfo::_create();
  pi.initialize(inPort, portName, type, ::sci::cca::core::PortType_ProvidesPort, properties);
  ports[portName] = pi;

  std::cerr << "scijump::Services_impl::addProvidesPort_impl: " << portName << std::endl;
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.addProvidesPort)
}

/**
 *  Returns the complete list of the properties for a Port.  This
 * includes the properties defined when the port was registered
 * (these properties can be modified by the framework), two special
 * properties "cca.portName" and "cca.portType", and any other
 * properties that the framework wishes to disclose to the component.
 * The framework may also choose to provide only the subset of input
 * properties (i.e. from addProvidesPort/registerUsesPort) that it
 * will honor.      
 */
::gov::cca::TypeMap
scijump::BabelServices_impl::getPortProperties_impl (
  /* in */const ::std::string& name ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.getPortProperties)
  // Insert-Code-Here {scijump.BabelServices.getPortProperties} (getPortProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelServices.getPortProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPortProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelServices.getPortProperties)
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.getPortProperties)
}

/**
 *  Notifies the framework that a previously exposed Port is no longer 
 * available for use. The Port being removed must exist
 * until this call returns, or a CCAException may occur.
 * @param portName The name of a provided Port.
 * @exception PortNotDefined. In general, the framework will not dictate 
 * when the component chooses to stop offering services.
 */
void
scijump::BabelServices_impl::removeProvidesPort_impl (
  /* in */const ::std::string& portName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.removeProvidesPort)

  Guard guard(lock_ports);

  PortMap::iterator iter = ports.find(portName);
  if ( iter == ports.end() ) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_PortNotDefined);
    ex.setNote("Port [" + portName + "] does not exist");
    ex.add(__FILE__, __LINE__, "unregisterUsesPort");
    throw ex;
  }

  // disconnect users or should user port do that only?
  ports.erase(iter);

  // DO-NOT-DELETE splicer.end(scijump.BabelServices.removeProvidesPort)
}

/**
 *  
 * Get a reference to the component to which this 
 * Services object belongs. 
 */
::gov::cca::ComponentID
scijump::BabelServices_impl::getComponentID_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.getComponentID)

  // maybe this should be in a separate class?
  scijump::BabelComponentInfo ci = scijump::BabelComponentInfo::_create();
  ci.initialize(selfInstanceName, selfClassName, framework, *this, selfProperties);
  return ::sidl::babel_cast< ::gov::cca::ComponentID>(ci);

  // DO-NOT-DELETE splicer.end(scijump.BabelServices.getComponentID)
}

/**
 *  Obtain a callback for component destruction.
 * @param callBack an object that implements the ComponentRelease
 * interface that will be called when the component is to be destroyed.
 * 
 * Register a callback to be executed when the component is going
 * to be destroyed.  During this callback, the Services object passed
 * through setServices will still be valid, but after all such
 * callbacks are made for a specific component, subsequent usage
 * of the Services object is not allowed/is undefined.
 */
void
scijump::BabelServices_impl::registerForRelease_impl (
  /* in */::gov::cca::ComponentRelease& callBack ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelServices.registerForRelease)
  // Insert-Code-Here {scijump.BabelServices.registerForRelease} (registerForRelease method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelServices.registerForRelease)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "registerForRelease");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelServices.registerForRelease)
  // DO-NOT-DELETE splicer.end(scijump.BabelServices.registerForRelease)
}


// DO-NOT-DELETE splicer.begin(scijump.BabelServices._misc)
// Insert-Code-Here {scijump.BabelServices._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.BabelServices._misc)


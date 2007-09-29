// 
// File:          scijump_BabelPortInfo_Impl.cxx
// Symbol:        scijump.BabelPortInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.BabelPortInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_BabelPortInfo_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_core_NotInitializedException_hxx
#include "sci_cca_core_NotInitializedException.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_sci_cca_core_PortType_hxx
#include "sci_cca_core_PortType.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo._includes)

#include "scijump_CCAException.hxx"
#include "scijump_TypeMap.hxx"

#include <iostream>

using namespace SCIRun;

// DO-NOT-DELETE splicer.end(scijump.BabelPortInfo._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::BabelPortInfo_impl::BabelPortInfo_impl() : StubBase(reinterpret_cast< 
  void*>(::scijump::BabelPortInfo::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo._ctor2)
  // Insert-Code-Here {scijump.BabelPortInfo._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo._ctor2)
}

// user defined constructor
void scijump::BabelPortInfo_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo._ctor)
  portType = ::sci::cca::core::PortType_UsesPort;
  lock = new SCIRun::Mutex("BabelPortInfo lock");
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo._ctor)
}

// user defined destructor
void scijump::BabelPortInfo_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo._dtor)

  delete lock;

  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo._dtor)
}

// static class initializer
void scijump::BabelPortInfo_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo._load)
  // Insert-Code-Here {scijump.BabelPortInfo._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::BabelPortInfo_impl::initialize_impl (
  /* in */::gov::cca::Port& port,
  /* in */const ::std::string& name,
  /* in */const ::std::string& className,
  /* in */::sci::cca::core::PortType portType,
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.initialize)

  // check for initialize?

  this->port = port;
  this->portType = portType;
  this->name = name;
  this->className = className;
  this->useCount = 0;

  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.initialize)
}

/**
 * Method:  initialize[Uses]
 */
void
scijump::BabelPortInfo_impl::initialize_impl (
  /* in */const ::std::string& name,
  /* in */const ::std::string& className,
  /* in */::sci::cca::core::PortType portType,
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.initializeUses)
  this->portType = portType;
  this->name = name;
  this->className = className;
  this->useCount = 0;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.initializeUses)
}

/**
 * Method:  connect[]
 */
bool
scijump::BabelPortInfo_impl::connect_impl (
  /* in */::sci::cca::core::PortInfo& to ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.connect)

  // TODO: document connect functions

  if ( to._is_nil() ) {
    std::cerr << "Attempting to connect to null port." << std::endl;
    return false;
  }
  if ( !canConnectTo(to) ) {
    std::cerr << "Connect test failed." << std::endl;
    return false;
  }

  if (portType == ::sci::cca::core::PortType_UsesPort
      && to.getPortType() == ::sci::cca::core::PortType_ProvidesPort) {
    Guard guard(lock);
    connections.push_back(to);
  } else {
    return to.connect(*this);
  }
  return true;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.connect)
}

/**
 * Method:  disconnect[]
 */
bool
scijump::BabelPortInfo_impl::disconnect_impl (
  /* in */::sci::cca::core::PortInfo& peer ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.disconnect)
  if (peer._is_nil()) return false;

  if (portType != ::sci::cca::core::PortType_UsesPort) {
    // warn?
    //std::cerr << "disconnect can be called only by user" << std::endl;
    return false;
  }

  Guard guard(lock);

  std::vector< ::sci::cca::core::PortInfo>::iterator iter;
  for (iter = connections.begin(); iter < connections.end(); iter++) {
    ::sci::cca::core::PortInfo cur = *iter;
    if (peer.isSame(cur)) {
      connections.erase(iter);
      return true;
    }
  }
  return false;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.disconnect)
}

/**
 * Method:  available[]
 */
bool
scijump::BabelPortInfo_impl::available_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.available)
  return portType == ::sci::cca::core::PortType_ProvidesPort || connections.size() == 0;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.available)
}

/**
 * Method:  canConnectTo[]
 */
bool
scijump::BabelPortInfo_impl::canConnectTo_impl (
  /* in */::sci::cca::core::PortInfo& toPortInfo ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.canConnectTo)

  return toPortInfo._not_nil()
    && available()
    && toPortInfo.available()
    && className == toPortInfo.getClass()
    && portType != toPortInfo.getPortType();

  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.canConnectTo)
}

/**
 * Method:  isConnected[]
 */
bool
scijump::BabelPortInfo_impl::isConnected_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.isConnected)
  return connections.size() > 0;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.isConnected)
}

/**
 * Method:  inUse[]
 */
bool
scijump::BabelPortInfo_impl::inUse_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.inUse)
  return useCount > 0;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.inUse)
}

/**
 * Method:  numOfConnections[]
 */
int32_t
scijump::BabelPortInfo_impl::numOfConnections_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.numOfConnections)
  return connections.size();
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.numOfConnections)
}

/**
 * Method:  getProperties[]
 */
::gov::cca::TypeMap
scijump::BabelPortInfo_impl::getProperties_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.getProperties)

  if (properties._is_nil())
    properties = scijump::TypeMap::_create();

  return properties;

  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.getProperties)
}

/**
 * Method:  setProperties[]
 */
void
scijump::BabelPortInfo_impl::setProperties_impl (
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.setProperties)
  // Insert-Code-Here {scijump.BabelPortInfo.setProperties} (setProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelPortInfo.setProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "setProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelPortInfo.setProperties)
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.setProperties)
}

/**
 * Method:  getPort[]
 */
::gov::cca::Port
scijump::BabelPortInfo_impl::getPort_impl () 
// throws:
//     ::sci::cca::core::NotInitializedException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.getPort)
  if (portType == ::sci::cca::core::PortType_ProvidesPort && port._is_nil()) {
    ::sci::cca::core::NotInitializedException ex = ::sci::cca::core::NotInitializedException::_create();
    ex.setNote("port has not been initialized");
    ex.add(__FILE__, __LINE__, "getPort");
    throw ex;
  }

  // throw exception for UsesPort or just allow null port?

  return port;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.getPort)
}

/**
 * Method:  getPeer[]
 */
::sci::cca::core::PortInfo
scijump::BabelPortInfo_impl::getPeer_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.getPeer)

  if ( portType == ::sci::cca::core::PortType_ProvidesPort || connections.size() == 0 ) {
    //throw CCAException::create("Port ["+name+"] Not Connected");
    ::gov::cca::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Port [" + name + "] is not connected");
    ex.add(__FILE__, __LINE__, "getPort");
    throw ex;
  }
  return connections[0];

  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.getPeer)
}

/**
 * Method:  getPortType[]
 */
::sci::cca::core::PortType
scijump::BabelPortInfo_impl::getPortType_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.getPortType)
  return portType;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.getPortType)
}

/**
 * Method:  getName[]
 */
::std::string
scijump::BabelPortInfo_impl::getName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.getName)
  return name;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.getName)
}

/**
 * Method:  getClass[]
 */
::std::string
scijump::BabelPortInfo_impl::getClass_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.getClass)
  return className;
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.getClass)
}

/**
 * Method:  incrementUseCount[]
 */
void
scijump::BabelPortInfo_impl::incrementUseCount_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.incrementUseCount)

  // lock this!
  Guard guard(lock);
  ++useCount;

  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.incrementUseCount)
}

/**
 * Method:  decrementUseCount[]
 */
bool
scijump::BabelPortInfo_impl::decrementUseCount_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.decrementUseCount)

  Guard guard(lock);
  if (useCount == 0) return false;

  --useCount;
  return true;

  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.decrementUseCount)
}

/**
 * Method:  invalidate[]
 */
void
scijump::BabelPortInfo_impl::invalidate_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo.invalidate)
  // Insert-Code-Here {scijump.BabelPortInfo.invalidate} (invalidate method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelPortInfo.invalidate)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "invalidate");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelPortInfo.invalidate)
  // DO-NOT-DELETE splicer.end(scijump.BabelPortInfo.invalidate)
}


// DO-NOT-DELETE splicer.begin(scijump.BabelPortInfo._misc)
// Insert-Code-Here {scijump.BabelPortInfo._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.BabelPortInfo._misc)


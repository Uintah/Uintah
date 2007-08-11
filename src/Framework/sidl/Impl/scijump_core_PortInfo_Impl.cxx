// 
// File:          scijump_core_PortInfo_Impl.cxx
// Symbol:        scijump.core.PortInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.core.PortInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_core_PortInfo_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._includes)

#include "scijump_CCAException.hxx"
#include "scijump_TypeMap.hxx"

#include <iostream>

// Insert-Code-Here {scijump.core.PortInfo._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.core.PortInfo._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::core::PortInfo_impl::PortInfo_impl() : StubBase(reinterpret_cast< 
  void*>(::scijump::core::PortInfo::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._ctor2)
  // Insert-Code-Here {scijump.core.PortInfo._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo._ctor2)
}

// user defined constructor
void scijump::core::PortInfo_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._ctor)
  portType = ::sci::cca::core::PortType_UsesPort;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo._ctor)
}

// user defined destructor
void scijump::core::PortInfo_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._dtor)
  // Insert-Code-Here {scijump.core.PortInfo._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo._dtor)
}

// static class initializer
void scijump::core::PortInfo_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._load)
  // Insert-Code-Here {scijump.core.PortInfo._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::core::PortInfo_impl::initialize_impl (
  /* in */::gov::cca::Port& port,
  /* in */const ::std::string& name,
  /* in */const ::std::string& className,
  /* in */::sci::cca::core::PortType portType,
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.initialize)

  // check for initialize?

  this->port = port;
  this->portType = portType;
  this->name = name;
  this->className = className;
  this->useCount = 0;

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.initialize)
}

/**
 * Method:  initialize[Uses]
 */
void
scijump::core::PortInfo_impl::initialize_impl (
  /* in */const ::std::string& name,
  /* in */const ::std::string& className,
  /* in */::sci::cca::core::PortType portType,
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.initializeUses)
  this->portType = portType;
  this->name = name;
  this->className = className;
  this->useCount = 0;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.initializeUses)
}

/**
 * Method:  connect[]
 */
bool
scijump::core::PortInfo_impl::connect_impl (
  /* in */::sci::cca::core::PortInfo& to ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.connect)

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
    //Guard guard(&lock);
    // lock this code!
    connections.push_back(to);
  } else {
    return to.connect(*this);
  }
  return true;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.connect)
}

/**
 * Method:  disconnect[]
 */
bool
scijump::core::PortInfo_impl::disconnect_impl (
  /* in */::sci::cca::core::PortInfo& peer ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.disconnect)
  // Insert-Code-Here {scijump.core.PortInfo.disconnect} (disconnect method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.disconnect)
//   ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
//   ex.setNote("This method has not been implemented");
//   ex.add(__FILE__, __LINE__, "disconnect");
//   throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.disconnect)

    if (peer._is_nil()) return false;

    if (portType != ::sci::cca::core::PortType_UsesPort) {
      // warn?
      //std::cerr << "disconnect can be called only by user" << std::endl;
      return false;
    }

    //Guard guard(&lock);
    // lock this code!

    std::vector< ::sci::cca::core::PortInfo>::iterator iter;
    for (iter = connections.begin(); iter < connections.end(); iter++) {
      ::sci::cca::core::PortInfo cur = *iter;
      if (peer.isSame(cur)) {
        connections.erase(iter);
        return true;
      }
    }
    return false;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.disconnect)
}

/**
 * Method:  available[]
 */
bool
scijump::core::PortInfo_impl::available_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.available)

  return portType == ::sci::cca::core::PortType_ProvidesPort || connections.size() == 0;

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.available)
}

/**
 * Method:  canConnectTo[]
 */
bool
scijump::core::PortInfo_impl::canConnectTo_impl (
  /* in */::sci::cca::core::PortInfo& toPortInfo ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.canConnectTo)

  return toPortInfo._not_nil()
    && available()
    && toPortInfo.available()
    && className == toPortInfo.getClass()
    && portType != toPortInfo.getPortType();

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.canConnectTo)
}

/**
 * Method:  isConnected[]
 */
bool
scijump::core::PortInfo_impl::isConnected_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.isConnected)
  return connections.size() > 0;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.isConnected)
}

/**
 * Method:  inUse[]
 */
bool
scijump::core::PortInfo_impl::inUse_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.inUse)

  return useCount > 0;

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.inUse)
}

/**
 * Method:  numOfConnections[]
 */
int32_t
scijump::core::PortInfo_impl::numOfConnections_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.numOfConnections)

  return connections.size();

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.numOfConnections)
}

/**
 * Method:  getProperties[]
 */
::gov::cca::TypeMap
scijump::core::PortInfo_impl::getProperties_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getProperties)
  if (properties._is_nil())
    properties = scijump::TypeMap::_create();

  return properties;

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getProperties)
}

/**
 * Method:  getPort[]
 */
::gov::cca::Port
scijump::core::PortInfo_impl::getPort_impl () 
// throws:
//     ::sci::cca::core::NotInitializedException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getPort)
  if (portType == ::sci::cca::core::PortType_ProvidesPort && port._is_nil()) {
    ::sci::cca::core::NotInitializedException ex = ::sci::cca::core::NotInitializedException::_create();
    ex.setNote("port has not been initialized");
    ex.add(__FILE__, __LINE__, "getPort");
    throw ex;
  }

  // throw exception for UsesPort or just allow null port?

  return port;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getPort)
}

/**
 * Method:  getPeer[]
 */
::sci::cca::core::PortInfo
scijump::core::PortInfo_impl::getPeer_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getPeer)

  if ( portType == ::sci::cca::core::PortType_ProvidesPort || connections.size() == 0 ) {
    //throw CCAException::create("Port ["+name+"] Not Connected");
    ::gov::cca::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Port [" + name + "] is not connected");
    ex.add(__FILE__, __LINE__, "getPort");
    throw ex;
  }
  return connections[0];

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getPeer)
}

/**
 * Method:  getPortType[]
 */
::sci::cca::core::PortType
scijump::core::PortInfo_impl::getPortType_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getPortType)
  return portType;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getPortType)
}

/**
 * Method:  getName[]
 */
::std::string
scijump::core::PortInfo_impl::getName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getName)
  return name;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getName)
}

/**
 * Method:  getClass[]
 */
::std::string
scijump::core::PortInfo_impl::getClass_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getClass)
  return className;
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getClass)
}

/**
 * Method:  incrementUseCount[]
 */
void
scijump::core::PortInfo_impl::incrementUseCount_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.incrementUseCount)

  // lock this!
  ++useCount;

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.incrementUseCount)
}

/**
 * Method:  decrementUseCount[]
 */
bool
scijump::core::PortInfo_impl::decrementUseCount_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.decrementUseCount)

  // lock this!
  if (useCount == 0) return false;

  --useCount;
  return true;

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.decrementUseCount)
}


// DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._misc)
// Insert-Code-Here {scijump.core.PortInfo._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.core.PortInfo._misc)


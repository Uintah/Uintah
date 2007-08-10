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
  // Insert-Code-Here {scijump.core.PortInfo._ctor} (constructor)
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
  /* in */::sci::cca::core::PortType portType,
  /* in */const ::std::string& name,
  /* in */const ::std::string& className ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.initialize)
  // Insert-Code-Here {scijump.core.PortInfo.initialize} (initialize method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.initialize)
//   ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
//   ex.setNote("This method has not been implemented");
//   ex.add(__FILE__, __LINE__, "initialize");
//   throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.initialize)

  // check for initialize?

  this->port = port;
  this->portType = portType;
  this->name = name;
  this->className = className;

  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.initialize)
}

/**
 * Method:  connect[]
 */
bool
scijump::core::PortInfo_impl::connect_impl (
  /* in */::sci::cca::core::PortInfo& portInfo ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.connect)
  // Insert-Code-Here {scijump.core.PortInfo.connect} (connect method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.connect)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "connect");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.connect)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.connect)
}

/**
 * Method:  disconnect[]
 */
bool
scijump::core::PortInfo_impl::disconnect_impl (
  /* in */::sci::cca::core::PortInfo& portInfo ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.disconnect)
  // Insert-Code-Here {scijump.core.PortInfo.disconnect} (disconnect method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.disconnect)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "disconnect");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.disconnect)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.disconnect)
}

/**
 * Method:  available[]
 */
bool
scijump::core::PortInfo_impl::available_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.available)
  // Insert-Code-Here {scijump.core.PortInfo.available} (available method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.available)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "available");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.available)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.available)
}

/**
 * Method:  canConnectTo[]
 */
bool
scijump::core::PortInfo_impl::canConnectTo_impl (
  /* in */::sci::cca::core::PortInfo& portInfo ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.canConnectTo)
  // Insert-Code-Here {scijump.core.PortInfo.canConnectTo} (canConnectTo method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.canConnectTo)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "canConnectTo");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.canConnectTo)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.canConnectTo)
}

/**
 * Method:  isConnected[]
 */
bool
scijump::core::PortInfo_impl::isConnected_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.isConnected)
  // Insert-Code-Here {scijump.core.PortInfo.isConnected} (isConnected method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.isConnected)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "isConnected");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.isConnected)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.isConnected)
}

/**
 * Method:  inUse[]
 */
bool
scijump::core::PortInfo_impl::inUse_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.inUse)
  // Insert-Code-Here {scijump.core.PortInfo.inUse} (inUse method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.inUse)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "inUse");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.inUse)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.inUse)
}

/**
 * Method:  numOfConnections[]
 */
int32_t
scijump::core::PortInfo_impl::numOfConnections_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.numOfConnections)
  // Insert-Code-Here {scijump.core.PortInfo.numOfConnections} (numOfConnections method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.numOfConnections)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "numOfConnections");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.numOfConnections)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.numOfConnections)
}

/**
 * Method:  getProperties[]
 */
::gov::cca::TypeMap
scijump::core::PortInfo_impl::getProperties_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getProperties)
  // Insert-Code-Here {scijump.core.PortInfo.getProperties} (getProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.getProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.getProperties)
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
  // Insert-Code-Here {scijump.core.PortInfo.getPort} (getPort method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.getPort)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPort");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.getPort)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getPort)
}

/**
 * Method:  getPeer[]
 */
::sci::cca::core::PortInfo
scijump::core::PortInfo_impl::getPeer_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getPeer)
  // Insert-Code-Here {scijump.core.PortInfo.getPeer} (getPeer method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.getPeer)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPeer");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.getPeer)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getPeer)
}

/**
 * Method:  getPortType[]
 */
::sci::cca::core::PortType
scijump::core::PortInfo_impl::getPortType_impl () 
// throws:
//     ::sci::cca::core::NotInitializedException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getPortType)
  // Insert-Code-Here {scijump.core.PortInfo.getPortType} (getPortType method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.getPortType)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPortType");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.getPortType)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getPortType)
}

/**
 * Method:  getName[]
 */
::std::string
scijump::core::PortInfo_impl::getName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getName)
  // Insert-Code-Here {scijump.core.PortInfo.getName} (getName method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.getName)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getName");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.getName)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getName)
}

/**
 * Method:  getClass[]
 */
::std::string
scijump::core::PortInfo_impl::getClass_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.getClass)
  // Insert-Code-Here {scijump.core.PortInfo.getClass} (getClass method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.getClass)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getClass");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.getClass)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.getClass)
}

/**
 * Method:  incrementUseCount[]
 */
void
scijump::core::PortInfo_impl::incrementUseCount_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.incrementUseCount)
  // Insert-Code-Here {scijump.core.PortInfo.incrementUseCount} (incrementUseCount method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.incrementUseCount)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "incrementUseCount");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.incrementUseCount)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.incrementUseCount)
}

/**
 * Method:  decrementUseCount[]
 */
bool
scijump::core::PortInfo_impl::decrementUseCount_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.PortInfo.decrementUseCount)
  // Insert-Code-Here {scijump.core.PortInfo.decrementUseCount} (decrementUseCount method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.PortInfo.decrementUseCount)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "decrementUseCount");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.PortInfo.decrementUseCount)
  // DO-NOT-DELETE splicer.end(scijump.core.PortInfo.decrementUseCount)
}


// DO-NOT-DELETE splicer.begin(scijump.core.PortInfo._misc)
// Insert-Code-Here {scijump.core.PortInfo._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.core.PortInfo._misc)


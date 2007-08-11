// 
// File:          scijump_core_ServiceInfo_Impl.cxx
// Symbol:        scijump.core.ServiceInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.core.ServiceInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_core_ServiceInfo_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sci_cca_core_NotInitializedException_hxx
#include "sci_cca_core_NotInitializedException.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._includes)
// Insert-Code-Here {scijump.core.ServiceInfo._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::core::ServiceInfo_impl::ServiceInfo_impl() : StubBase(
  reinterpret_cast< void*>(::scijump::core::ServiceInfo::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._ctor2)
  // Insert-Code-Here {scijump.core.ServiceInfo._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._ctor2)
}

// user defined constructor
void scijump::core::ServiceInfo_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._ctor)
  // Insert-Code-Here {scijump.core.ServiceInfo._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._ctor)
}

// user defined destructor
void scijump::core::ServiceInfo_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._dtor)
  // Insert-Code-Here {scijump.core.ServiceInfo._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._dtor)
}

// static class initializer
void scijump::core::ServiceInfo_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._load)
  // Insert-Code-Here {scijump.core.ServiceInfo._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::core::ServiceInfo_impl::initialize_impl (
  /* in */const ::std::string& serviceName,
  /* in */::sci::cca::core::PortInfo& servicePort,
  /* in */::sci::cca::core::PortInfo& requesterPort ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo.initialize)
  this->serviceName = serviceName;
  this->servicePort = servicePort;
  this->requesterPort = requesterPort;
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo.initialize)
}

/**
 * Method:  getServiceName[]
 */
::std::string
scijump::core::ServiceInfo_impl::getServiceName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo.getServiceName)
  return serviceName;
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo.getServiceName)
}

/**
 * Method:  getServicePortName[]
 */
::std::string
scijump::core::ServiceInfo_impl::getServicePortName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo.getServicePortName)
  return servicePort.getName();
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo.getServicePortName)
}

/**
 * Method:  getServicePort[]
 */
::sci::cca::core::PortInfo
scijump::core::ServiceInfo_impl::getServicePort_impl () 
// throws:
//     ::sci::cca::core::NotInitializedException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo.getServicePort)
  return servicePort;
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo.getServicePort)
}

/**
 * Method:  getRequesterPortName[]
 */
::std::string
scijump::core::ServiceInfo_impl::getRequesterPortName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo.getRequesterPortName)
  return requesterPort.getName();
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo.getRequesterPortName)
}

/**
 * Method:  getRequesterPort[]
 */
::sci::cca::core::PortInfo
scijump::core::ServiceInfo_impl::getRequesterPort_impl () 
// throws:
//     ::sci::cca::core::NotInitializedException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo.getRequesterPort)
  return requesterPort;
  // DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo.getRequesterPort)
}


// DO-NOT-DELETE splicer.begin(scijump.core.ServiceInfo._misc)
// Insert-Code-Here {scijump.core.ServiceInfo._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.core.ServiceInfo._misc)


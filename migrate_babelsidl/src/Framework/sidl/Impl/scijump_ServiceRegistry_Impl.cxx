// 
// File:          scijump_ServiceRegistry_Impl.cxx
// Symbol:        scijump.ServiceRegistry-v0.2.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for scijump.ServiceRegistry
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_ServiceRegistry_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
#endif
#ifndef included_gov_cca_ports_ServiceProvider_hxx
#include "gov_cca_ports_ServiceProvider.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_scijump_core_ServiceInfo_hxx
#include "scijump_core_ServiceInfo.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._includes)
#include <iostream>
#include <scijump_BabelPortInfo.hxx>
// DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::ServiceRegistry_impl::ServiceRegistry_impl() : StubBase(
  reinterpret_cast< void*>(::scijump::ServiceRegistry::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._ctor2)
  // Insert-Code-Here {scijump.ServiceRegistry._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._ctor2)
}

// user defined constructor
void scijump::ServiceRegistry_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._ctor)
  // Insert-Code-Here {scijump.ServiceRegistry._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._ctor)
}

// user defined destructor
void scijump::ServiceRegistry_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._dtor)
  // Insert-Code-Here {scijump.ServiceRegistry._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._dtor)
}

// static class initializer
void scijump::ServiceRegistry_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._load)
  // Insert-Code-Here {scijump.ServiceRegistry._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._load)
}

// user defined static methods:
/**
 * Method:  create[]
 */
::sci::cca::core::FrameworkService
scijump::ServiceRegistry_impl::create_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry.create)
  scijump::ServiceRegistry sr = scijump::ServiceRegistry::_create();
  sr.initialize(framework);
  return sr;
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry.create)
}


// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::ServiceRegistry_impl::initialize_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry.initialize)
  this->framework = framework;
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry.initialize)
}

/**
 * Method:  getService[]
 */
::scijump::core::ServiceInfo
scijump::ServiceRegistry_impl::getService_impl (
  /* in */const ::std::string& serviceName,
  /* in */::sci::cca::core::PortInfo& requesterPort ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry.getService)
  portMap::iterator iter = singletons.find(serviceName);
  if (iter == singletons.end()) {
    scijump::core::ServiceInfo si = scijump::core::ServiceInfo::_create();
    si.initialize("EMPTY",NULL,NULL);
    return si;
  }

  gov::cca::Port service = iter->second;  
  scijump::BabelPortInfo servicePort = scijump::BabelPortInfo::_create();
  servicePort.initialize(service, serviceName, serviceName, ::sci::cca::core::PortType_ProvidesPort, NULL);
  if (! requesterPort.connect(servicePort)) {
    // TODO: throw exception?
    std::cerr << "Could not connect " << serviceName << " service." << std::endl;
  }
  
  scijump::core::ServiceInfo si = scijump::core::ServiceInfo::_create();
  si.initialize(serviceName, servicePort, requesterPort);
  return si;
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry.getService)
}

/**
 * Add a ServiceProvider that can be asked to produce service Port's
 * for other components to use subsequently.
 * True means success. False means that for some reason, the
 * provider isn't going to function. Possibly another server is doing
 * the job.
 */
bool
scijump::ServiceRegistry_impl::addService_impl (
  /* in */const ::std::string& serviceType,
  /* in */::gov::cca::ports::ServiceProvider& portProvider ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry.addService)
  // Insert-Code-Here {scijump.ServiceRegistry.addService} (addService method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.ServiceRegistry.addService)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "addService");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.ServiceRegistry.addService)
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry.addService)
}

/**
 *  Add a "reusable" service gov.cca.Port for other components to use 
 * subsequently.
 */
bool
scijump::ServiceRegistry_impl::addSingletonService_impl (
  /* in */const ::std::string& serviceType,
  /* in */::gov::cca::Port& server ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry.addSingletonService)
  portMap::iterator iter = singletons.find(serviceType);
  if (iter != singletons.end())
    return false;

  std::cerr << "Service '" << serviceType << "' now in ServiceRegistry\n";
  singletons[serviceType] = server;
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry.addSingletonService)
}

/**
 *  Inform the framework that this service Port is no longer to
 * be used, subsequent to this call. 
 */
void
scijump::ServiceRegistry_impl::removeService_impl (
  /* in */const ::std::string& serviceType ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry.removeService)
  portMap::iterator iter = singletons.find(serviceType);
  if (iter != singletons.end()) {
    singletons.erase(iter);
  }
  // DO-NOT-DELETE splicer.end(scijump.ServiceRegistry.removeService)
}


// DO-NOT-DELETE splicer.begin(scijump.ServiceRegistry._misc)
// DO-NOT-DELETE splicer.end(scijump.ServiceRegistry._misc)


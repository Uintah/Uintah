// 
// File:          scijump_core_FrameworkServiceFactory_Impl.cxx
// Symbol:        scijump.core.FrameworkServiceFactory-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.core.FrameworkServiceFactory
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_core_FrameworkServiceFactory_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._includes)

//#include "sci_cca.hxx"

#include <iostream>

// Insert-Code-Here {scijump.core.FrameworkServiceFactory._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::core::FrameworkServiceFactory_impl::FrameworkServiceFactory_impl() : 
  StubBase(reinterpret_cast< void*>(
  ::scijump::core::FrameworkServiceFactory::_wrapObj(reinterpret_cast< void*>(
  this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._ctor2)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._ctor2)
}

// user defined constructor
void scijump::core::FrameworkServiceFactory_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._ctor)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._ctor)
}

// user defined destructor
void scijump::core::FrameworkServiceFactory_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._dtor)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._dtor)
}

// static class initializer
void scijump::core::FrameworkServiceFactory_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._load)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::core::FrameworkServiceFactory_impl::initialize_impl (
  /* in */void* internalFactoryImpl ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory.initialize)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory.initialize} (initialize method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.core.FrameworkServiceFactory.initialize)
//   ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
//   ex.setNote("This method has not been implemented");
//   ex.add(__FILE__, __LINE__, "initialize");
//   throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.core.FrameworkServiceFactory.initialize)

  factory = static_cast<ServiceFactory*>(internalFactoryImpl);
  if (! factory) {
    ::sci::cca::core::NotInitializedException ex = ::sci::cca::core::NotInitializedException::_create();
    ex.setNote("factory pointer is null");
    ex.add(__FILE__, __LINE__, "getService");
    throw ex;
  }
  serviceName = factory->getName();

  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory.initialize)
}

/**
 * Method:  getName[]
 */
::std::string
scijump::core::FrameworkServiceFactory_impl::getName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory.getName)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory.getName} (getName method)

  return serviceName;
  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory.getName)
}

/**
 * Method:  getService[]
 */
::sci::cca::core::PortInfo
scijump::core::FrameworkServiceFactory_impl::getService_impl (
  /* in */const ::std::string& serviceName,
  /* in */::gov::cca::ComponentID& requester ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory.getService)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory.getService} (getService method)

  if (! factory) {
    ::sci::cca::core::NotInitializedException ex = ::sci::cca::core::NotInitializedException::_create();
    ex.setNote("factory pointer is null");
    ex.add(__FILE__, __LINE__, "getService");
    throw ex;
  }

  return factory->get(serviceName, requester);
  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory.getService)
}

/**
 * Method:  releaseService[]
 */
void
scijump::core::FrameworkServiceFactory_impl::releaseService_impl (
  /* in */const ::std::string& portName ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory.releaseService)
  // Insert-Code-Here {scijump.core.FrameworkServiceFactory.releaseService} (releaseService method)

  if (! factory) {
    ::sci::cca::core::NotInitializedException ex = ::sci::cca::core::NotInitializedException::_create();
    ex.setNote("factory pointer is null");
    ex.add(__FILE__, __LINE__, "getService");
    throw ex;
  }

  return factory->release(portName);

  // DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory.releaseService)
}


// DO-NOT-DELETE splicer.begin(scijump.core.FrameworkServiceFactory._misc)

// Insert-Code-Here {scijump.core.FrameworkServiceFactory._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.core.FrameworkServiceFactory._misc)


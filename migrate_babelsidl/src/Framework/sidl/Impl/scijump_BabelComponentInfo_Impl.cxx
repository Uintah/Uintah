// 
// File:          scijump_BabelComponentInfo_Impl.cxx
// Symbol:        scijump.BabelComponentInfo-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.BabelComponentInfo
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_BabelComponentInfo_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_scijump_BabelServices_hxx
#include "scijump_BabelServices.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._includes)
#include "scijump.hxx"

#include <Core/Thread/Guard.h>

using namespace SCIRun;
// DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::BabelComponentInfo_impl::BabelComponentInfo_impl() : StubBase(
  reinterpret_cast< void*>(::scijump::BabelComponentInfo::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._ctor2)
  // Insert-Code-Here {scijump.BabelComponentInfo._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._ctor2)
}

// user defined constructor
void scijump::BabelComponentInfo_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._ctor)
  valid = false;
  lock = new Mutex("BabelComponentInfo lock");
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._ctor)
}

// user defined destructor
void scijump::BabelComponentInfo_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._dtor)
  delete lock;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._dtor)
}

// static class initializer
void scijump::BabelComponentInfo_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._load)
  // Insert-Code-Here {scijump.BabelComponentInfo._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::BabelComponentInfo_impl::initialize_impl (
  /* in */const ::std::string& instanceName,
  /* in */const ::std::string& className,
  /* in */::sci::cca::AbstractFramework& framework,
  /* in */::gov::cca::Component& component,
  /* in */::scijump::BabelServices& services,
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.initialize)

  Guard g(lock);
  this->instanceName = instanceName;
  this->className = className;
  this->component = component;
  this->services = services;
  this->framework = framework;
  if (properties._is_nil()) {
    this->properties = scijump::TypeMap::_create();
  } else {
    this->properties = properties;
  }
  this->valid = true;

  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.initialize)
}

/**
 * Method:  initialize[Full]
 */
void
scijump::BabelComponentInfo_impl::initialize_impl (
  /* in */const ::std::string& instanceName,
  /* in */const ::std::string& className,
  /* in */::sci::cca::AbstractFramework& framework,
  /* in */::gov::cca::Component& component,
  /* in */::scijump::BabelServices& services,
  /* in */::gov::cca::TypeMap& properties,
  /* in */const ::std::string& serialization ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.initializeFull)

  Guard g(lock);
  this->instanceName = instanceName;
  this->className = className;
  this->component = component;
  this->services = services;
  this->framework = framework;
  this->properties = properties;
  this->valid = true;

  // set serialization

  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.initializeFull)
}

/**
 * Method:  initialize[ComponentImage]
 */
void
scijump::BabelComponentInfo_impl::initialize_impl (
  /* in */const ::std::string& instanceName,
  /* in */const ::std::string& className,
  /* in */::sci::cca::AbstractFramework& framework,
  /* in */::scijump::BabelServices& services,
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.initializeComponentImage)

  Guard g(lock);
  this->instanceName = instanceName;
  this->className = className;
  this->framework = framework;
  this->services = services;
  this->properties = properties;
  this->valid = true;

  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.initializeComponentImage)
}

/**
 * Method:  setSerialization[]
 */
void
scijump::BabelComponentInfo_impl::setSerialization_impl (
  /* in */const ::std::string& serialization ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.setSerialization)
  // Insert-Code-Here {scijump.BabelComponentInfo.setSerialization} (setSerialization method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelComponentInfo.setSerialization)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "setSerialization");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelComponentInfo.setSerialization)
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.setSerialization)
}

/**
 * Method:  getFramework[]
 */
::sci::cca::AbstractFramework
scijump::BabelComponentInfo_impl::getFramework_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getFramework)
  return framework;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getFramework)
}

/**
 * Method:  getComponent[]
 */
::gov::cca::Component
scijump::BabelComponentInfo_impl::getComponent_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getComponent)
  return component;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getComponent)
}

/**
 * Method:  getServices[]
 */
::gov::cca::Services
scijump::BabelComponentInfo_impl::getServices_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getServices)
  return services;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getServices)
}

/**
 * Method:  getPorts[]
 */
::sidl::array< ::sci::cca::core::PortInfo>
scijump::BabelComponentInfo_impl::getPorts_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getPorts)
  // Insert-Code-Here {scijump.BabelComponentInfo.getPorts} (getPorts method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelComponentInfo.getPorts)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPorts");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelComponentInfo.getPorts)
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getPorts)
}

/**
 * Method:  getPortInfo[]
 */
::sci::cca::core::PortInfo
scijump::BabelComponentInfo_impl::getPortInfo_impl (
  /* in */const ::std::string& portName ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getPortInfo)
  return services.getPortInfo(portName);
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getPortInfo)
}

/**
 * Method:  getClassName[]
 */
::std::string
scijump::BabelComponentInfo_impl::getClassName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getClassName)
  if (className.size() < 1 || ! valid) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "getClassName");
    throw ex;
  } 
  return className;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getClassName)
}

/**
 * Method:  getProperties[]
 */
::gov::cca::TypeMap
scijump::BabelComponentInfo_impl::getProperties_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getProperties)
  return properties;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getProperties)
}

/**
 * Method:  setProperties[]
 */
void
scijump::BabelComponentInfo_impl::setProperties_impl (
  /* in */::gov::cca::TypeMap& properties ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.setProperties)
  Guard g(lock);
  if (properties._not_nil()) {
    this->properties = properties;
  }
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.setProperties)
}

/**
 * Method:  callReleaseCallback[]
 */
bool
scijump::BabelComponentInfo_impl::callReleaseCallback_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.callReleaseCallback)
  Guard g(lock);
  if (services._is_nil() || ! valid) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "callReleaseCallback");
    throw ex;
  }
  bool ret = services.callReleaseCallback();
  return ret;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.callReleaseCallback)
}

/**
 * Method:  invalidate[]
 */
void
scijump::BabelComponentInfo_impl::invalidate_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.invalidate)
  Guard g(lock);
  component = 0;
  properties = 0;
  services = 0;
  framework = 0;
  instanceName.clear();
  className.clear();
  valid = false;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.invalidate)
}

/**
 *  
 * Returns the instance name provided in 
 * <code>BuilderService.createInstance()</code>
 * or in 
 * <code>AbstractFramework.getServices()</code>.
 * @throws CCAException if <code>ComponentID</code> is invalid
 */
::std::string
scijump::BabelComponentInfo_impl::getInstanceName_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getInstanceName)
  if (instanceName.size() < 1 || ! valid) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "getInstanceName");
    throw ex;
  } 
  return instanceName;
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getInstanceName)
}

/**
 * Returns a framework specific serialization of the ComponentID.
 * @throws CCAException if <code>ComponentID</code> is
 * invalid.
 */
::std::string
scijump::BabelComponentInfo_impl::getSerialization_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo.getSerialization)
  // Insert-Code-Here {scijump.BabelComponentInfo.getSerialization} (getSerialization method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BabelComponentInfo.getSerialization)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getSerialization");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BabelComponentInfo.getSerialization)
  // DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo.getSerialization)
}


// DO-NOT-DELETE splicer.begin(scijump.BabelComponentInfo._misc)
// Insert-Code-Here {scijump.BabelComponentInfo._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.BabelComponentInfo._misc)


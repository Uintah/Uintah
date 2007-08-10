// 
// File:          scijump_SCIJumpFramework_Impl.cxx
// Symbol:        scijump.SCIJumpFramework-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.SCIJumpFramework
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_SCIJumpFramework_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_AbstractFramework_hxx
#include "gov_cca_AbstractFramework.hxx"
#endif
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
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
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._includes)

#include "scijump.hxx"

#include <Framework/Core/SingletonServiceFactory.h>

#include <iostream>

// Insert-Code-Here {scijump.SCIJumpFramework._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::SCIJumpFramework_impl::SCIJumpFramework_impl() : StubBase(
  reinterpret_cast< void*>(::scijump::SCIJumpFramework::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._ctor2)
  // Insert-Code-Here {scijump.SCIJumpFramework._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._ctor2)
}

// user defined constructor
void scijump::SCIJumpFramework_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._ctor)

  initFrameworkServices();

  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._ctor)
}

// user defined destructor
void scijump::SCIJumpFramework_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._dtor)

  destroyFrameworkServices();

  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._dtor)
}

// static class initializer
void scijump::SCIJumpFramework_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._load)
  // Insert-Code-Here {scijump.SCIJumpFramework._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Registers the slave framework with the master framework. Intended to be called
 * only by the representative slave framework process.
 * @param size Total number of parallel slave frameworks.
 * @param slaveURLs Urls of the slave framework.
 * @param slaveName Name of the slave resource.
 * @return A positive number or zero if framework was registered
 * successfully, negative number on error.
 */
int32_t
scijump::SCIJumpFramework_impl::registerLoader_impl (
  /* in */const ::std::string& slaveName,
  /* in array<string> */::sidl::array< ::std::string>& slaveURLs ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.registerLoader)
  // Insert-Code-Here {scijump.SCIJumpFramework.registerLoader} (registerLoader method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.SCIJumpFramework.registerLoader)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "registerLoader");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.SCIJumpFramework.registerLoader)
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.registerLoader)
}

/**
 * Method:  unregisterLoader[]
 */
int32_t
scijump::SCIJumpFramework_impl::unregisterLoader_impl (
  /* in */const ::std::string& slaveName ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.unregisterLoader)
  // Insert-Code-Here {scijump.SCIJumpFramework.unregisterLoader} (unregisterLoader method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.SCIJumpFramework.unregisterLoader)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "unregisterLoader");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.SCIJumpFramework.unregisterLoader)
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.unregisterLoader)
}

/**
 *  
 * Create an empty TypeMap. Presumably this would be used in 
 * an ensuing call to <code>getServices()</code>. The "normal" method of
 * creating typemaps is found in the <code>Services</code> interface. It
 * is duplicated here to break the "chicken and egg" problem.
 */
::gov::cca::TypeMap
scijump::SCIJumpFramework_impl::createTypeMap_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.createTypeMap)
  return scijump::TypeMap::_create();
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.createTypeMap)
}

/**
 *  
 * Retrieve a Services handle to the underlying framework. 
 * This interface effectively causes the calling program to 
 * appear as the image of a component inside the framework.
 * This method may be called any number of times
 * with different arguments, creating a new component image 
 * each time. 
 * The only proper method to destroy a Services obtained 
 * from this interface is to pass it to releaseServices.
 * 
 * @param selfInstanceName the Component instance name,
 * as it will appear in the framework.
 * 
 * @param selfClassName the Component type of the 
 * calling program, as it will appear in the framework. 
 * 
 * @param selfProperties (which can be null) the properties 
 * of the component image to appear. 
 * 
 * @throws CCAException in the event that selfInstanceName 
 * is already in use by another component.
 * 
 * @return  A Services object that pertains to the
 * image of the this component. This is identical
 * to the object passed into Component.setServices() 
 * when a component is created.
 */
::gov::cca::Services
scijump::SCIJumpFramework_impl::getServices_impl (
  /* in */const ::std::string& selfInstanceName,
  /* in */const ::std::string& selfClassName,
  /* in */::gov::cca::TypeMap& selfProperties ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.getServices)
  // Insert-Code-Here {scijump.SCIJumpFramework.getServices} (getServices method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.SCIJumpFramework.getServices)
//   ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
//   ex.setNote("This method has not been implemented");
//   ex.add(__FILE__, __LINE__, "getServices");
//   throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.SCIJumpFramework.getServices)

  if (services.find(selfInstanceName) != services.end()) {
    // throw CCAException: selfInstanceName is already in use by another component
    ::gov::cca::CCAException ex = scijump::CCAException::_create();
    std::string note("Instance " + selfInstanceName + " is already in use by another component");
    ex.setNote(note);
    ex.add(__FILE__, __LINE__, "getServices");
    throw ex;
  }
  scijump::Services s = scijump::Services::_create();
  // TODO: need to create a component for the caller...
  services[selfInstanceName] = s;
  return s;

  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.getServices)
}

/**
 *  
 * Inform framework that the <code>Services</code> handle is no longer needed by the 
 * caller and that the reference to its component image is to be
 * deleted from the context of the underlying framework. This invalidates
 * any <code>ComponentID</code>'s or <code>ConnectionID</code>'s associated 
 * with the given <code>Services</code>' component image. 
 * 
 * @param services The result of getServices earlier obtained.
 * 
 * @throws CCAException if the <code>Services</code>
 * handle has already been released or is otherwise rendered invalid 
 * or was not obtained from <code>getServices()</code>.
 */
void
scijump::SCIJumpFramework_impl::releaseServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.releaseServices)
  // Insert-Code-Here {scijump.SCIJumpFramework.releaseServices} (releaseServices method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.SCIJumpFramework.releaseServices)
//   ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
//   ex.setNote("This method has not been implemented");
//   ex.add(__FILE__, __LINE__, "releaseServices");
//   throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.SCIJumpFramework.releaseServices)

  ServicesMap::iterator pos = this->services.end();
  std::string instanceName;
  for (ServicesMap::iterator iter = this->services.begin(); iter != this->services.end(); iter++) {
    ::gov::cca::Services cur = iter->second;
    if ( cur.isSame(services) ) {
      pos = iter;
      instanceName = iter->first;
      break;
    }
  }

  if (pos == this->services.end()) {
    // warn?
    return;
  }
  // destroy associated component etc.
  this->services.erase(pos);

  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.releaseServices)
}

/**
 *  
 * Tell the framework it is no longer needed and to clean up after itself. 
 * @throws CCAException if the framework has already been shutdown.
 */
void
scijump::SCIJumpFramework_impl::shutdownFramework_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.shutdownFramework)
  // Insert-Code-Here {scijump.SCIJumpFramework.shutdownFramework} (shutdownFramework method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.SCIJumpFramework.shutdownFramework)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "shutdownFramework");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.SCIJumpFramework.shutdownFramework)
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.shutdownFramework)
}

/**
 *  
 * Creates a new framework instance based on the same underlying 
 * framework implementation. This does not copy the existing 
 * framework, nor are any of the user-instantiated components in
 * the original framework available in the newly created 
 * <code>AbstractFramework</code>. 
 * 
 * @throws CCAException when one of the following conditions occur:
 * 
 * (1)the AbstractFramework previously had shutdownFramework() called on it, or 
 * (2)the underlying framework implementation does not permit creation 
 * of another instance.	 
 */
::gov::cca::AbstractFramework
scijump::SCIJumpFramework_impl::createEmptyFramework_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.createEmptyFramework)

  return scijump::SCIJumpFramework::_create();

  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.createEmptyFramework)
}


// DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._misc)
// Insert-Code-Here {scijump.SCIJumpFramework._misc} (miscellaneous code)

void
scijump::SCIJumpFramework_impl::initFrameworkServices()
{
  scijump::core::FrameworkServiceFactory bsf = scijump::core::FrameworkServiceFactory::_create();
  bsf.initialize( new scijump::core::SingletonServiceFactory<scijump::BuilderService>(*this, "cca.BuilderService") );
  addFrameworkService(bsf, this->frameworkServices);

  scijump::core::FrameworkServiceFactory psf = scijump::core::FrameworkServiceFactory::_create();
  psf.initialize( new scijump::core::SingletonServiceFactory<scijump::PublisherEventService>(*this, "cca.PublisherEventService") );
  addFrameworkService(psf, this->frameworkServices);

  scijump::core::FrameworkServiceFactory ssf = scijump::core::FrameworkServiceFactory::_create();
  psf.initialize( new scijump::core::SingletonServiceFactory<scijump::SubscriberEventService>(*this, "cca.SubscriberEventService") );
  addFrameworkService(psf, this->frameworkServices);
}

void
scijump::SCIJumpFramework_impl::destroyFrameworkServices()
{
  frameworkServices.clear();
}

bool
scijump::SCIJumpFramework_impl::addFrameworkService(
                                                    ::scijump::core::FrameworkServiceFactory& factory,
                                                    FrameworkServiceMap& frameworkServices)
{
  std::string n = factory.getName();
  FrameworkServiceMap::iterator iter = frameworkServices.find(n);
  if (iter != frameworkServices.end())
    return false;

  frameworkServices[n] = factory;
  //std::cerr << "addFrameworkService(..) " << n << " done" << std::endl;
  return true;
}

// DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._misc)


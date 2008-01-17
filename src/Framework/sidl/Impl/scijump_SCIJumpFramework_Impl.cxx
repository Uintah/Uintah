// 
// File:          scijump_SCIJumpFramework_Impl.cxx
// Symbol:        scijump.SCIJumpFramework-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
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
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_ConnectionID_hxx
#include "gov_cca_ConnectionID.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_core_ComponentInfo_hxx
#include "sci_cca_core_ComponentInfo.hxx"
#endif
#ifndef included_sci_cca_core_PortInfo_hxx
#include "sci_cca_core_PortInfo.hxx"
#endif
#ifndef included_sci_cca_core_ServiceInfo_hxx
#include "sci_cca_core_ServiceInfo.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._includes)
#include "scijump.hxx"

#include <sci_defs/framework_defs.h>
#include <Framework/Core/SingletonServiceFactory.h>
#include <Core/Thread/Guard.h>
#include <Core/Util/Assert.h>

#include <iostream>
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
  lock_components = new SCIRun::Mutex("SCIRunFramework::components lock");
  lock_connections = new SCIRun::Mutex("SCIRunFramework::connections lock");

  initFrameworkServices();
  // replace with component model factories - see Plume
  bcm = new BabelComponentModel(*this);
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._ctor)
}

// user defined destructor
void scijump::SCIJumpFramework_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework._dtor)
  frameworkServices.clear();

  delete lock_components;
  delete lock_connections;
  delete bcm;
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
 * Method:  isFrameworkService[]
 */
bool
scijump::SCIJumpFramework_impl::isFrameworkService_impl (
  /* in */const ::std::string& name ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.isFrameworkService)

  // from the Plume framework

  FrameworkServiceMap::const_iterator iter = frameworkServices.find(name);
#if FWK_DEBUG
  std::cerr << "scijump::SCIJumpFramework_impl::isFrameworkService_impl(..) service=" << iter->first << std::endl;
#endif
  return iter != frameworkServices.end();
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.isFrameworkService)
}

/**
 * Method:  getFrameworkService[]
 */
::scijump::core::ServiceInfo
scijump::SCIJumpFramework_impl::getFrameworkService_impl (
  /* in */const ::std::string& serviceName,
  /* in */::sci::cca::core::PortInfo& requesterPort ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.getFrameworkService)

  // from the Plume framework

  // lock this code!
  //Guard guard(&service_lock);

  FrameworkServiceMap::const_iterator iter = frameworkServices.find(serviceName);
  if ( iter == frameworkServices.end() )
    return scijump::core::ServiceInfo::_create();
#if FWK_DEBUG
  std::cerr << "scijump::SCIJumpFramework_impl::getFrameworkService_impl(..) service=" << iter->first << std::endl;
#endif
  // get a port from the service
  ::sci::cca::core::FrameworkServiceFactory f = iter->second;
  ::sci::cca::core::PortInfo servicePort = f.getService(serviceName);

  // connect the requester port and the service ports (service port is always the provider)
  if (! requesterPort.connect(servicePort)) {
    // TODO: throw exception?
    std::cerr << "Could not connect " << serviceName << " service." << std::endl;
  }

  // do we need to maintain a reference to this connection ?
  scijump::core::ServiceInfo si = scijump::core::ServiceInfo::_create();
  si.initialize(serviceName, servicePort, requesterPort);
  return si;
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.getFrameworkService)
}

/**
 * Method:  releaseFrameworkService[]
 */
void
scijump::SCIJumpFramework_impl::releaseFrameworkService_impl (
  /* in */::sci::cca::core::ServiceInfo& info ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.releaseFrameworkService)

  // from the Plume framework

  // lock this code!
  //Guard guard(&service_lock);

  // disconnect
  ::sci::cca::core::PortInfo servicePort = info.getServicePort();
  ::sci::cca::core::PortInfo requesterPort = info.getRequesterPort();
  requesterPort.disconnect(servicePort);

  // release service
  FrameworkServiceMap::const_iterator iter = frameworkServices.find(info.getServiceName());
  //if ( iter == frameworkServices.end() )
  //  return;

  ::sci::cca::core::FrameworkServiceFactory f = iter->second;
  f.releaseService( info.getServicePortName() );
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.releaseFrameworkService)
}

/**
 *  Eliminates the component instance ``cid'' from the scope of the
 * framework.  The ``timeout'' parameter specifies the maximum allowable
 * wait time for this operation.  A timeout of 0 leaves the wait time up to
 * the framework.  If the destroy operation is not completed in the maximum
 * allowed number of seconds, or the referenced component does not exist,
 * then a CCAException is thrown.
 * 
 * Like createComponentInstance, this method is only intended to be called
 * by the BuilderService class.  It searches the list of registered
 * components (compIDs) for the matching component ID, unregisters it, finds
 * the correct ComponentModel for the type, then calls
 * ComponentModel::destroyInstance to properly destroy the component. 
 */
void
scijump::SCIJumpFramework_impl::destroyComponentInstance_impl (
  /* in */::gov::cca::ComponentID& cid,
  /* in */float timeout ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.destroyComponentInstance)

  if (timeout != 0) {
    std::cerr << "WARNING: timeout ignored for now." << std::endl;
  }

  ::sci::cca::core::ComponentInfo ci = ::sidl::babel_cast< ::sci::cca::core::ComponentInfo>(cid);
  if (ci._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid component object");
    ex.add(__FILE__, __LINE__, "destroyComponentInstance");
    throw ex;
  }
  ComponentInstanceMap::iterator iter = components.find(ci.getInstanceName());
  if (iter != components.end()) {
    // just BabelComponentModel for now
    bcm->destroyInstance(iter->second);
    components.erase(iter);
  }
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.destroyComponentInstance)
}

/**
 *  Creates an instance of the component defined by the string ``type'',
 * which must uniquely define the type of the component.  The component
 * instance is given the name ``name''.   If the instance name is not
 * specified (i.e. the method is passed an empty string), then the component
 * will be assigned a unique name automatically.
 * 
 * This method is ``semi-private'' and intended to be called only by the
 * BuilderService class.  It works by searching the list of ComponentModels
 * (the ivar \em models) for a matching registered type, and then calling
 * the createInstance method on the appropriate ComponentModel object. 
 */
::gov::cca::ComponentID
scijump::SCIJumpFramework_impl::createComponentInstance_impl (
  /* in */const ::std::string& name,
  /* in */const ::std::string& className,
  /* in */::gov::cca::TypeMap& tm ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.createComponentInstance)

  // See if the type is of the form:
  //   model:name
  // If so, extract the model and look up that component specifically.
  // Otherwise, look at all models for that component

  std::string type = className;

  /*
  unsigned int firstColon = type.find(':');
  if (firstColon < type.size()) {
    std::string modelName = type.substr(0, firstColon);
    type = type.substr(firstColon+1);

    for (std::vector<ComponentModel*>::iterator iter = models.begin();
         iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if (model->getPrefixName() == modelName) {
        mod = model;
        break;
      }
    }
  } else {
    int count = 0;
    for (std::vector<ComponentModel*>::iterator iter = models.begin();
         iter != models.end(); iter++) {
      ComponentModel* model = *iter;
      if (model->haveComponent(type)) {
        count++;
        mod = model;
      }
    }
    if (count > 1) {
      throw sci::cca::CCAException::pointer(
        new CCAException("More than one component model wants to build " + type));
    }
  }
  if (!mod) {
    throw sci::cca::CCAException::pointer(
      new CCAException("Unknown class name for " + name));
  }
  */

  scijump::BabelComponentInfo bci = scijump::BabelComponentInfo::_create();
  bcm->createInstance(bci, name, type, tm);
#if FWK_DEBUG
  ASSERT(bci._not_nil());
#else
  if (bci._is_nil()) {
    // throw exception?
    std::cerr << "Error: failed to create BabelComponentInfo" << std::endl;
    return NULL;
  }
#endif
  {
    Guard guard(lock_components);

#if 0 // code from Plume for component class factories
    /*
    {
      Guard guard(&factory_lock);
      ComponentClassFactoryMap::iterator factory = factories.find(className);
      if ( factory == factories.end() )
        throw CCAException::create("Can not create a component of type ["+className+"]: no factory");

      // this may throw a cca exception.
      // do not catch it.

      services = factory->second->create( pointer(this), instanceName, properties);
    }
    */
#endif

    sci::cca::core::ComponentInfo ci;
    ci = bci;
    components[name] = ci;
  }

#if 0 // from Plume
  /*
  // we should not init the component until the CoreServices is in the component array
  // otherwise we might get a race condition.
  try {
    if ( !services->getComponent().isNull() )
      services->getComponent()->setServices(services);
  } catch (...) {
    std::cerr << "component crashed during set services\n";
    Guard guard(&component_lock);
    ComponentMap::iterator iter = components.find(instanceName);
    components.erase(iter);
    services = 0;
  }

  return services;
  */
#endif

  return bci;
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.createComponentInstance)
}

/**
 * Method:  getComponentInstance[]
 */
::gov::cca::ComponentID
scijump::SCIJumpFramework_impl::getComponentInstance_impl (
  /* in */const ::std::string& name ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.getComponentInstance)
  Guard g(lock_components);
  ComponentInstanceMap::iterator iter = components.find(name);
  if( iter != components.end() ) {
    ::sci::cca::core::ComponentInfo ci = iter->second;
    return ::sidl::babel_cast< ::gov::cca::ComponentID>(ci);
  }
  return 0; 
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.getComponentInstance)
}

/**
 * Method:  getComponentInstances[]
 */
::sidl::array< ::gov::cca::ComponentID>
scijump::SCIJumpFramework_impl::getComponentInstances_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.getComponentInstances)
  Guard g(lock_components);
  int num_components = components.size();
  ::sidl::array< ::gov::cca::ComponentID> cids = 
      ::sidl::array< ::gov::cca::ComponentID>::create1d(num_components);
  ComponentInstanceMap::iterator iter = components.begin();
  for(int i=0; i < num_components; i++,iter++) {
    cids.set(i,babel_cast< ::gov::cca::ComponentID>(iter->second));
  }

  return cids;
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.getComponentInstances)
}

/**
 * Method:  createConnectionInstance[]
 */
::gov::cca::ConnectionID
scijump::SCIJumpFramework_impl::createConnectionInstance_impl (
  /* in */::sci::cca::core::ComponentInfo& user,
  /* in */::sci::cca::core::ComponentInfo& provider,
  /* in */const ::std::string& userPortName,
  /* in */const ::std::string& providerPortName,
  /* in */::gov::cca::TypeMap& tm ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.createConnectionInstance)
  Guard g(lock_connections);

  scijump::BabelConnectionInfo ci = scijump::BabelConnectionInfo::_create();
  ci.initialize(user, provider, userPortName, providerPortName, tm);

  if (find_if(connections.begin(), connections.end(), ConnectionInfo_eq(ci)) != connections.end()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Connection already created");
    ex.add(__FILE__, __LINE__, "createConnectionInstance");
    throw ex;
  }

  connections.push_back(ci);
  return ci;
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.createConnectionInstance)
}

/**
 * Method:  destroyConnectionInstance[]
 */
void
scijump::SCIJumpFramework_impl::destroyConnectionInstance_impl (
  /* in */::gov::cca::ConnectionID& connID ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.destroyConnectionInstance)
  ::sci::cca::core::ConnectionInfo ci = ::sidl::babel_cast< ::sci::cca::core::ConnectionInfo>(connID);
  if (ci._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_Unexpected);
    ex.setNote("Invalid connection object");
    ex.add(__FILE__, __LINE__, "destroyConnectionInstance");
    throw ex;
  }

  ConnectionList::iterator iter = find_if(connections.begin(), connections.end(), ConnectionInfo_eq(ci));
  if (iter != connections.end()) {
    iter->invalidate();
    connections.erase(iter);
  }
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.destroyConnectionInstance)
}

/**
 * Method:  getConnectionInstances[]
 */
::sidl::array< ::gov::cca::ConnectionID>
scijump::SCIJumpFramework_impl::getConnectionInstances_impl (
  /* in array<gov.cca.ComponentID> */::sidl::array< ::gov::cca::ComponentID>& 
    componentList ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.getConnectionInstances)
  Guard g(lock_connections);
  ::sidl::array< ::gov::cca::ConnectionID> connids = 
      ::sidl::array< ::gov::cca::ConnectionID>::create1d(connections.size());
  int conn_ctr = 0;
  ConnectionList::iterator iter = connections.begin();
  for(; iter != connections.end(); iter++) {
    gov::cca::ComponentID provider = iter->getProvider();
    gov::cca::ComponentID user = iter->getUser();
    for(int j=0; j<componentList.length(); j++) {
      if((user.getInstanceName() == componentList.get(j).getInstanceName()) ||
	 (provider.getInstanceName() ==  componentList.get(j).getInstanceName())) {

	connids.set(conn_ctr,::sidl::babel_cast< ::gov::cca::ConnectionID>(*iter));
	conn_ctr++;
	break;
      }
    }
  }

  //Copying the array in order to reduce its size... Better way?
  ::sidl::array< ::gov::cca::ConnectionID> n_connids = 
      ::sidl::array< ::gov::cca::ConnectionID>::create1d(conn_ctr);
  for(int i=0; i< conn_ctr; i++) {
    n_connids.set(i,connids.get(i));
  }

  return n_connids;
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.getConnectionInstances)
}

/**
 *  Adds a description of a component instance (class ComponentInstance) to
 * the list of active components.  The component instance description
 * includes the component type name, the instance name, and the pointer to
 * the allocated component.  When a \em name conflicts with an existing
 * registered component instance name, this method will automatically append
 * an integer to create a new, unique instance name.
 */
::std::string
scijump::SCIJumpFramework_impl::getUniqueName_impl (
  /* in */const ::std::string& name ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.SCIJumpFramework.getUniqueName)
  std::string goodname = name;
  int count = 0;
  Guard g(lock_components);
  while (components.find(goodname) != components.end()) {
    std::ostringstream newname;
    newname << name << "_" << count++;
    goodname = newname.str();
  }
  return goodname;
  // DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework.getUniqueName)
}

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

  if (services.find(selfInstanceName) != services.end()) {
    // throw CCAException: selfInstanceName is already in use by another component
    ::gov::cca::CCAException ex = scijump::CCAException::_create();
    std::string note("Instance " + selfInstanceName + " is already in use by another component");
    ex.setNote(note);
    ex.add(__FILE__, __LINE__, "getServices");
    throw ex;
  }
  scijump::BabelServices s = scijump::BabelServices::_create();
  s.initialize(*this, selfInstanceName, selfClassName, selfProperties);

  // TODO: need to create a (CCA) component for the caller...

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
  // TODO: destroy associated component etc.
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
  services.clear();
  components.clear();
  connections.clear();
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

  scijump::core::FrameworkServiceFactory esf = scijump::core::FrameworkServiceFactory::_create();
  esf.initialize( new scijump::core::SingletonServiceFactory<scijump::EventService>(*this, "cca.EventService") );
  addFrameworkService(esf, this->frameworkServices);

  scijump::core::FrameworkServiceFactory alf = scijump::core::FrameworkServiceFactory::_create();
  alf.initialize( new scijump::core::SingletonServiceFactory<scijump::ApplicationLoaderService>(*this, "cca.ApplicationLoaderService") );
  addFrameworkService(alf, this->frameworkServices);
}

bool
scijump::SCIJumpFramework_impl::addFrameworkService(
                                                    ::scijump::core::FrameworkServiceFactory& factory,
                                                    FrameworkServiceMap& frameworkServices)
{
  // lock
  std::string n = factory.getName();
  FrameworkServiceMap::iterator iter = frameworkServices.find(n);
  if (iter != frameworkServices.end())
    return false;

  frameworkServices[n] = factory;
  //std::cerr << "addFrameworkService(..) " << n << " done" << std::endl;
  return true;
}

bool
scijump::SCIJumpFramework_impl::removeFrameworkService(const std::string& serviceName, FrameworkServiceMap& frameworkServices)
{
  // lock
  FrameworkServiceMap::iterator iter = frameworkServices.find(serviceName);
  if (iter != frameworkServices.end()) {
    frameworkServices.erase(iter);
    return true;
  }

  return false;
}

#if 0
/** Adds a description of a component instance (class ComponentInstance) to
    the list of active components.  The component instance description
    includes the component type name, the instance name, and the pointer to
    the allocated component.  When a \em name conflicts with an existing
    registered component instance name, this method will automatically append
    an integer to create a new, unique instance name.*/
//gov::cca::ComponentID 
//scijump::SCIJumpFramework_impl::registerComponent(ComponentInstance *ci, const std::string& name)
//{
//}

/** Removes a component instance description from the list of active
    framework components.  Returns the pointer to the component description
    that was successfully unregistered. */
/*
ComponentInstance* 
scijump::SCIJumpFramework_impl::unregisterComponent(const std::string& instanceName) 
{
  SCIRun::Guard g1(lock_activeInstances);

  ComponentInstanceMap::iterator found = activeInstances.find(instanceName);
  if (found != activeInstances.end()) {
    ComponentInstance *ci = found->second;
    activeInstances.erase(found);
    return ci;
  } else {
    std::cerr << "Error: component instance " << instanceName << " not found!" << std::endl;;
    return 0;
  }
}
*/
#endif
// DO-NOT-DELETE splicer.end(scijump.SCIJumpFramework._misc)


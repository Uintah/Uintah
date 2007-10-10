// 
// File:          scijump_BuilderService_Impl.cxx
// Symbol:        scijump.BuilderService-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.BuilderService
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_BuilderService_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_ConnectionID_hxx
#include "gov_cca_ConnectionID.hxx"
#endif
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_Event_hxx
#include "sci_cca_Event.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.BuilderService._includes)

#include "sidl.hxx"
#include "scijump.hxx"
#include "sci_cca.hxx"

#include <sci_defs/framework_defs.h>
#include <Core/Util/Assert.h>

#include <iostream>

// Insert-Code-Here {scijump.BuilderService._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.BuilderService._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::BuilderService_impl::BuilderService_impl() : StubBase(
  reinterpret_cast< void*>(::scijump::BuilderService::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService._ctor2)
  // Insert-Code-Here {scijump.BuilderService._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService._ctor2)
}

// user defined constructor
void scijump::BuilderService_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService._ctor)
  // Insert-Code-Here {scijump.BuilderService._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService._ctor)
}

// user defined destructor
void scijump::BuilderService_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService._dtor)
  // Insert-Code-Here {scijump.BuilderService._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService._dtor)
}

// static class initializer
void scijump::BuilderService_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService._load)
  // Insert-Code-Here {scijump.BuilderService._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService._load)
}

// user defined static methods:
/**
 * Method:  create[]
 */
::sci::cca::core::FrameworkService
scijump::BuilderService_impl::create_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.create)
  scijump::BuilderService bs = scijump::BuilderService::_create();
  bs.initialize(framework);

  return bs;
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.create)
}


// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::BuilderService_impl::initialize_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.initialize)
  this->framework = ::sidl::babel_cast<scijump::SCIJumpFramework>(framework);
  serviceInfo = ::scijump::core::ServiceInfo::_create();

  ::scijump::BabelPortInfo pi = ::scijump::BabelPortInfo::_create();
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.initialize)
}

/**
 *  Get available ports in c2 that can be connected to port1 of c1. 
 */
::sidl::array< ::std::string>
scijump::BuilderService_impl::getCompatiblePortList_impl (
  /* in */::gov::cca::ComponentID& c1,
  /* in */const ::std::string& port1,
  /* in */::gov::cca::ComponentID& c2 ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getCompatiblePortList)
  // Insert-Code-Here {scijump.BuilderService.getCompatiblePortList} (getCompatiblePortList method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.getCompatiblePortList)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getCompatiblePortList");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.getCompatiblePortList)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getCompatiblePortList)
}

/**
 * TODO: document getBridgeablePortList
 * @param cid1
 * @param port1
 * @param  cid2
 */
::sidl::array< ::std::string>
scijump::BuilderService_impl::getBridgeablePortList_impl (
  /* in */::gov::cca::ComponentID& cid1,
  /* in */const ::std::string& port1,
  /* in */::gov::cca::ComponentID& cid2 ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getBridgeablePortList)
  // Insert-Code-Here {scijump.BuilderService.getBridgeablePortList} (getBridgeablePortList method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.getBridgeablePortList)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getBridgeablePortList");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.getBridgeablePortList)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getBridgeablePortList)
}

/**
 * Method:  generateBridge[]
 */
::std::string
scijump::BuilderService_impl::generateBridge_impl (
  /* in */::gov::cca::ComponentID& user,
  /* in */const ::std::string& usingPortName,
  /* in */::gov::cca::ComponentID& provider,
  /* in */const ::std::string& providingPortName ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.generateBridge)
  // Insert-Code-Here {scijump.BuilderService.generateBridge} (generateBridge method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.generateBridge)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "generateBridge");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.generateBridge)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.generateBridge)
}

/**
 * Creates an instance of a CCA component of the type defined by the 
 * string className.  The string classname uniquely defines the
 * "type" of the component, e.g.
 * doe.cca.Library.GaussianElmination. 
 * It has an instance name given by the string instanceName.
 * The instanceName may be empty (zero length) in which case
 * the instanceName will be assigned to the component automatically.
 * @throws CCAException If the Component className is unknown, or if the
 * instanceName has already been used, a CCAException is thrown.
 * @return A ComponentID corresponding to the created component. Destroying
 * the returned ID does not destroy the component; 
 * see destroyInstance instead.
 */
::gov::cca::ComponentID
scijump::BuilderService_impl::createInstance_impl (
  /* in */const ::std::string& instanceName,
  /* in */const ::std::string& className,
  /* in */::gov::cca::TypeMap& properties ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.createInstance)
  if (instanceName.size()) {
    if (framework.getComponentInstance(instanceName)._not_nil()) {
      scijump::CCAException ex = scijump::CCAException::_create();
      ex.setNote("Component instance name " + instanceName + " is not unique");
      ex.add(__FILE__, __LINE__, "createInstance");
      throw ex;
    }
    return framework.createComponentInstance(instanceName,className,properties);
  }

  return framework.createComponentInstance(framework.getUniqueName(className),className,properties);
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.createInstance)
}

/**
 *  
 * Get component list.
 * @return a ComponentID for each component currently created.
 */
::sidl::array< ::gov::cca::ComponentID>
scijump::BuilderService_impl::getComponentIDs_impl () 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getComponentIDs)
  return framework.getComponentInstances();
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getComponentIDs)
}

/**
 *  
 * Get property map for component.
 * @return the public properties associated with the component referred to by
 * ComponentID. 
 * @throws a CCAException if the ComponentID is invalid.
 */
::gov::cca::TypeMap
scijump::BuilderService_impl::getComponentProperties_impl (
  /* in */::gov::cca::ComponentID& cid ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getComponentProperties)
  std::cerr << "BuilderService::getComponentProperties is not implemented\n";
  return scijump::TypeMap::_create();
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getComponentProperties)
}

/**
 * Causes the framework implementation to associate the given properties 
 * with the component designated by cid. 
 * @throws CCAException if cid is invalid or if there is an attempted
 * change to a property locked by the framework implementation.
 */
void
scijump::BuilderService_impl::setComponentProperties_impl (
  /* in */::gov::cca::ComponentID& cid,
  /* in */::gov::cca::TypeMap& map ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.setComponentProperties)
  // Insert-Code-Here {scijump.BuilderService.setComponentProperties} (setComponentProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.setComponentProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "setComponentProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.setComponentProperties)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.setComponentProperties)
}

/**
 *  Get component id from stringified reference.
 * @return a ComponentID from the string produced by 
 * ComponentID.getSerialization(). 
 * @throws CCAException if the string does not represent the appropriate 
 * serialization of a ComponentID for the underlying framework.
 */
::gov::cca::ComponentID
scijump::BuilderService_impl::getDeserialization_impl (
  /* in */const ::std::string& s ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getDeserialization)
  // Insert-Code-Here {scijump.BuilderService.getDeserialization} (getDeserialization method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.getDeserialization)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getDeserialization");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.getDeserialization)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getDeserialization)
}

/**
 *  Get id from name by which it was created.
 * @return a ComponentID from the instance name of the component
 * produced by ComponentID.getInstanceName().
 * @throws CCAException if there is no component matching the 
 * given componentInstanceName.
 */
::gov::cca::ComponentID
scijump::BuilderService_impl::getComponentID_impl (
  /* in */const ::std::string& componentInstanceName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getComponentID)
  return framework.getComponentInstance(componentInstanceName);
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getComponentID)
}

/**
 *  
 * Eliminate the Component instance, from the scope of the framework.
 * @param toDie the component to be removed.
 * @param timeout the allowable wait; 0 means up to the framework.
 * @throws CCAException if toDie refers to an invalid component, or
 * if the operation takes longer than timeout seconds.
 */
void
scijump::BuilderService_impl::destroyInstance_impl (
  /* in */::gov::cca::ComponentID& toDie,
  /* in */float timeout ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.destroyInstance)

  // TODO: Need some sort of timer here, but a blocking wait is not a good thing.
  // Use a worker thread to do disconnect???

  if (timeout != 0) {
    std::cerr << "WARNING: timeout ignored for now." << std::endl;
  }

  framework.destroyComponentInstance(toDie, timeout);

  // DO-NOT-DELETE splicer.end(scijump.BuilderService.destroyInstance)
}

/**
 *  
 * Get the names of Port instances provided by the identified component.
 * @param cid the component.
 * @throws CCAException if cid refers to an invalid component.
 */
::sidl::array< ::std::string>
scijump::BuilderService_impl::getProvidedPortNames_impl (
  /* in */::gov::cca::ComponentID& cid ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getProvidedPortNames)
  // Insert-Code-Here {scijump.BuilderService.getProvidedPortNames} (getProvidedPortNames method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.getProvidedPortNames)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getProvidedPortNames");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.getProvidedPortNames)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getProvidedPortNames)
}

/**
 *  
 * Get the names of Port instances used by the identified component.
 * @param cid the component.
 * @throws CCAException if cid refers to an invalid component. 
 */
::sidl::array< ::std::string>
scijump::BuilderService_impl::getUsedPortNames_impl (
  /* in */::gov::cca::ComponentID& cid ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getUsedPortNames)
  // Insert-Code-Here {scijump.BuilderService.getUsedPortNames} (getUsedPortNames method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.getUsedPortNames)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getUsedPortNames");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.getUsedPortNames)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getUsedPortNames)
}

/**
 *  
 * Fetch map of Port properties exposed by the framework.
 * @return the public properties pertaining to the Port instance 
 * portname on the component referred to by cid. 
 * @throws CCAException when any one of the following conditions occur:<ul>
 * <li>portname is not a registered Port on the component indicated by cid,
 * <li>cid refers to an invalid component. </ul>
 */
::gov::cca::TypeMap
scijump::BuilderService_impl::getPortProperties_impl (
  /* in */::gov::cca::ComponentID& cid,
  /* in */const ::std::string& portName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getPortProperties)
  // Insert-Code-Here {scijump.BuilderService.getPortProperties} (getPortProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.getPortProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getPortProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.getPortProperties)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getPortProperties)
}

/**
 *  
 * Associates the properties given in map with the Port indicated by 
 * portname. The component must have a Port known by portname.
 * @throws CCAException if either cid or portname are
 * invalid, or if this a changed property is locked by 
 * the underlying framework or component.
 */
void
scijump::BuilderService_impl::setPortProperties_impl (
  /* in */::gov::cca::ComponentID& cid,
  /* in */const ::std::string& portName,
  /* in */::gov::cca::TypeMap& map ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.setPortProperties)
  // Insert-Code-Here {scijump.BuilderService.setPortProperties} (setPortProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.setPortProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "setPortProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.setPortProperties)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.setPortProperties)
}

/**
 * Creates a connection between ports on component user and 
 * component provider. Destroying the ConnectionID does not
 * cause a disconnection; for that, see disconnect().
 * @throws CCAException when any one of the following conditions occur:<ul>
 * <li>If either user or provider refer to an invalid component,
 * <li>If either usingPortName or providingPortName refer to a 
 * nonexistent Port on their respective component,
 * <li>If other-- In reality there are a lot of things that can go wrong 
 * with this operation, especially if the underlying connections 
 * involve networking.</ul>
 */
::gov::cca::ConnectionID
scijump::BuilderService_impl::connect_impl (
  /* in */::gov::cca::ComponentID& user,
  /* in */const ::std::string& usingPortName,
  /* in */::gov::cca::ComponentID& provider,
  /* in */const ::std::string& providingPortName ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.connect)
  ::sci::cca::core::ComponentInfo ciUser = ::sidl::babel_cast< ::sci::cca::core::ComponentInfo>(user);
  if (ciUser._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot connect: invalid user componentID");
    ex.add(__FILE__, __LINE__, "connect");
    throw ex;
  }
  ::sci::cca::core::PortInfo piUser = ciUser.getPortInfo(usingPortName);
  if (piUser._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_BadPortName);
    ex.setNote("Unknown port " + usingPortName);
    ex.add(__FILE__, __LINE__, "connect");
    throw ex;
  }

  ::sci::cca::core::ComponentInfo ciProvider = ::sidl::babel_cast< ::sci::cca::core::ComponentInfo>(provider);
  if (ciProvider._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot connect: invalid provider componentID");
    ex.add(__FILE__, __LINE__, "connect");
    throw ex;
  }
  ::sci::cca::core::PortInfo piProvider = ciProvider.getPortInfo(providingPortName);
  if (piProvider._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_BadPortName);
    ex.setNote("Unknown port " + providingPortName);
    ex.add(__FILE__, __LINE__, "connect");
    throw ex;
  }

  if (! piUser.connect(piProvider)) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot connect " + usingPortName + " with " + providingPortName);
    ex.add(__FILE__, __LINE__, "connect");
    throw ex;
  }

  ::gov::cca::TypeMap properties = scijump::TypeMap::_create();
  properties.putString("user", ciUser.getInstanceName());
  properties.putString("provider", ciUser.getInstanceName());
  properties.putString("uses port", usingPortName);
  properties.putString("provides port", providingPortName);
  // not bridging at the moment
  //properties.putBool("bridge", isBridge);

  ::gov::cca::ConnectionID cid = framework.createConnectionInstance(ciUser, ciProvider, usingPortName, providingPortName, properties);
  return cid;
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.connect)
}

/**
 *  Returns a list of connections as an array of 
 * handles. This will return all connections involving components 
 * in the given componentList of ComponentIDs. This
 * means that ConnectionID's will be returned even if only one 
 * of the participants in the connection appears in componentList.
 * 
 * @throws CCAException if any component in componentList is invalid.
 */
::sidl::array< ::gov::cca::ConnectionID>
scijump::BuilderService_impl::getConnectionIDs_impl (
  /* in array<gov.cca.ComponentID> */::sidl::array< ::gov::cca::ComponentID>& 
    componentList ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getConnectionIDs)
  return framework.getConnectionInstances(componentList);
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getConnectionIDs)
}

/**
 * Fetch property map of a connection.
 * @returns the properties for the given connection.
 * @throws CCAException if connID is invalid.
 */
::gov::cca::TypeMap
scijump::BuilderService_impl::getConnectionProperties_impl (
  /* in */::gov::cca::ConnectionID& connID ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.getConnectionProperties)
  // Insert-Code-Here {scijump.BuilderService.getConnectionProperties} (getConnectionProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.getConnectionProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "getConnectionProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.getConnectionProperties)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.getConnectionProperties)
}

/**
 *  Associates the properties with the connection.
 * @param map the source of the properties.
 * @param connID connection to receive property values.
 * @throws CCAException if connID is invalid, or if this changes 
 * a property locked by the underlying framework.
 */
void
scijump::BuilderService_impl::setConnectionProperties_impl (
  /* in */::gov::cca::ConnectionID& connID,
  /* in */::gov::cca::TypeMap& map ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.setConnectionProperties)
  // Insert-Code-Here {scijump.BuilderService.setConnectionProperties} (setConnectionProperties method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.setConnectionProperties)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "setConnectionProperties");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.setConnectionProperties)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.setConnectionProperties)
}

/**
 *  Disconnect the connection indicated by connID before the indicated
 * timeout in secs. Upon successful completion, connID and the connection
 * it represents become invalid. 
 * @param timeout the time in seconds to wait for a connection to close; 0
 * means to use the framework implementation default.
 * @param connID the connection to be broken.
 * @throws CCAException when any one of the following conditions occur: <ul>
 * <li>id refers to an invalid ConnectionID,
 * <li>timeout is exceeded, after which, if id was valid before 
 * disconnect() was invoked, it remains valid
 * </ul>
 */
void
scijump::BuilderService_impl::disconnect_impl (
  /* in */::gov::cca::ConnectionID& connID,
  /* in */float timeout ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.disconnect)

  // TODO: Need some sort of timer here, but a blocking wait is not a good thing.
  // Use a worker thread to do disconnect???

  if (timeout != 0) {
    std::cerr << "WARNING: timeout ignored for now." << std::endl;
  }

  ::gov::cca::ComponentID user = connID.getUser();
  ::gov::cca::ComponentID provider = connID.getProvider();
  std::string usingPortName = connID.getUserPortName();
  std::string providingPortName = connID.getProviderPortName();

  ::sci::cca::core::ComponentInfo ciUser = ::sidl::babel_cast< ::sci::cca::core::ComponentInfo>(user);
  if (ciUser._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot connect: invalid user componentID");
    ex.add(__FILE__, __LINE__, "disconnect");
    throw ex;
  }
  ::sci::cca::core::PortInfo piUser = ciUser.getPortInfo(usingPortName);
  if (piUser._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_BadPortName);
    ex.setNote("Unknown port " + usingPortName);
    ex.add(__FILE__, __LINE__, "disconnect");
    throw ex;
  }

  ::sci::cca::core::ComponentInfo ciProvider = ::sidl::babel_cast< ::sci::cca::core::ComponentInfo>(provider);
  if (ciProvider._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot connect: invalid provider componentID");
    ex.add(__FILE__, __LINE__, "disconnect");
    throw ex;
  }
  ::sci::cca::core::PortInfo piProvider = ciProvider.getPortInfo(providingPortName);
  if (piProvider._is_nil()) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.initialize(::gov::cca::CCAExceptionType_BadPortName);
    ex.setNote("Unknown port " + providingPortName);
    ex.add(__FILE__, __LINE__, "disconnect");
    throw ex;
  }

  if (! piUser.disconnect(piProvider)) {
    scijump::CCAException ex = scijump::CCAException::_create();
    ex.setNote("Cannot connect " + usingPortName + " with " + providingPortName);
    ex.add(__FILE__, __LINE__, "disconnect");
    throw ex;
  }

  framework.destroyConnectionInstance(connID);
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.disconnect)
}

/**
 *  Remove all connections between components id1 and id2 within 
 * the period of timeout secs. If id2 is null, then all connections 
 * to id1 are removed (within the period of timeout secs).
 * @throws CCAException when any one of the following conditions occur:<ul>
 * <li>id1 or id2 refer to an invalid ComponentID (other than id2 == null),
 * <li>The timeout period is exceeded before the disconnections can be made. 
 * </ul>
 */
void
scijump::BuilderService_impl::disconnectAll_impl (
  /* in */::gov::cca::ComponentID& id1,
  /* in */::gov::cca::ComponentID& id2,
  /* in */float timeout ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.disconnectAll)
  // Insert-Code-Here {scijump.BuilderService.disconnectAll} (disconnectAll method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.disconnectAll)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "disconnectAll");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.disconnectAll)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.disconnectAll)
}

/**
 *  This is where event processing by a listener takes place. This
 * is a call-back method that a topic subscriber implements and
 * gets called for each new event.
 * 
 * @topicName - The topic for which the Event was created and sent.
 * @theEvent - The payload.
 */
void
scijump::BuilderService_impl::processEvent_impl (
  /* in */const ::std::string& topicName,
  /* in */::sci::cca::Event& theEvent ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.BuilderService.processEvent)
  // Insert-Code-Here {scijump.BuilderService.processEvent} (processEvent method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.BuilderService.processEvent)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "processEvent");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.BuilderService.processEvent)
  // DO-NOT-DELETE splicer.end(scijump.BuilderService.processEvent)
}


// DO-NOT-DELETE splicer.begin(scijump.BuilderService._misc)

#if 0
  // Insert-Code-Here {scijump.BuilderService.getServiceInfo} (getServiceInfo method)
//   if (serviceInfo._is_nil()) {
//     ::sci::cca::core::NotInitializedException ex = ::sci::cca::core::NotInitializedException::_create();
//     ex.setNote("scijump::BuilderService_impl::ServiceInfo member is nil");
//     ex.add(__FILE__, __LINE__, "getServiceInfo");
//     throw ex;
//   }

//   return ::sidl::babel_cast< ::sci::cca::core::ServiceInfo>(serviceInfo);
#endif

// Insert-Code-Here {scijump.BuilderService._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.BuilderService._misc)


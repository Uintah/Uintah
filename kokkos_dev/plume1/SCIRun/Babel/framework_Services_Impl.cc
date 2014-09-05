//
//  For more information, please see: http://software.sci.utah.edu
// 
//  The MIT License
// 
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
// 
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
// 
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
// 
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
// 

// 
// File:          framework_Services_Impl.cc
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for framework.Services
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "framework_Services_Impl.hh"

// DO-NOT-DELETE splicer.begin(framework.Services._includes)
// Put additional includes or other arbitrary code here...
#include <iostream>
using namespace std;
// DO-NOT-DELETE splicer.end(framework.Services._includes)

// user-defined constructor.
void framework::Services_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.Services._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(framework.Services._ctor)
}

// user-defined destructor.
void framework::Services_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.Services._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(framework.Services._dtor)
}

// static class initializer.
void framework::Services_impl::_load() {
  // DO-NOT-DELETE splicer.begin(framework.Services._load)
  // Insert-Code-Here {framework.Services._load} (class initialization)
  // DO-NOT-DELETE splicer.end(framework.Services._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  getData[]
 */
void*
framework::Services_impl::getData ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(framework.Services.getData)
  return &ports;
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.Services.getData)
}

/**
 * Fetch a previously registered Port (defined by either 
 * addProvidePort or (more typically) registerUsesPort).  
 * @return Will return the Port (possibly waiting forever while
 * attempting to acquire it) or throw an exception. Does not return
 * NULL, even in the case where no connection has been made. 
 * If a Port is returned,
 * there is then a contract that the port will remain valid for use
 * by the caller until the port is released via releasePort(), or a 
 * Disconnect Event is successfully dispatched to the caller,
 * or a runtime exception (such as network failure) occurs during 
 * invocation of some function in the Port. 
 * <p>
 * Subtle interpretation: If the Component is not listening for
 * Disconnect events, then the framework has no clean way to
 * break the connection until after the component calls releasePort.
 * </p>
 * <p>The framework may go through some machinations to obtain
 *    the port, possibly involving an interactive user or network 
 *    queries, before giving up and throwing an exception.
 * </p>
 * 
 * @param portName The previously registered or provide port which
 * 	   the component now wants to use.
 * @exception CCAException with the following types: NotConnected, PortNotDefined, 
 *                NetworkError, OutOfMemory.
 */
::gov::cca::Port
framework::Services_impl::getPort (
  /* in */ const ::std::string& portName ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.getPort)
  return getPortNonblocking(portName);
  // DO-NOT-DELETE splicer.end(framework.Services.getPort)
}

/**
 * Get a previously registered Port (defined by
 * either addProvide or registerUses) and return that
 * Port if it is available immediately (already connected
 * without further connection machinations).
 * There is an contract that the
 * port will remain valid per the description of getPort.
 * @return The named port, if it exists and is connected or self-provided,
 * 	      or NULL if it is registered and is not yet connected. Does not
 * 	      return if the Port is neither registered nor provided, but rather
 * 	      throws an exception.
 * @param portName registered or provided port that
 * 	     the component now wants to use.
 * @exception CCAException with the following types: PortNotDefined, OutOfMemory.
 */
::gov::cca::Port
framework::Services_impl::getPortNonblocking (
  /* in */ const ::std::string& portName ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.getPortNonblocking)
  map<string, PortInstance*>::iterator iter = ports.find(portName);
  if (iter == ports.end()) {
    return 0;
  }

  BabelPortInstance* pr = dynamic_cast<BabelPortInstance*> (iter->second);
  if (pr->porttype == BabelPortInstance::Provides) {
    cerr<<"Cannot call getPort on a Provides port"<<endl;
    //throw CCAException("Cannot call getPort on a Provides port");
  }

  pr->incrementUseCount();
  if (pr->connections.size() != 1) {
      return 0;
  }
  BabelPortInstance *pi=dynamic_cast<BabelPortInstance*> (pr->getPeer());
  return pi->port;
  // DO-NOT-DELETE splicer.end(framework.Services.getPortNonblocking)
}

/**
 * Notifies the framework that this component is finished 
 * using the previously fetched Port that is named.     
 * The releasePort() method calls should be paired with 
 * getPort() method calls; however, an extra call to releasePort()
 * for the same name may (is not required to) generate an exception.
 * Calls to release ports which are not defined or have never be fetched
 * with one of the getPort functions generate exceptions.
 * @param portName The name of a port.
 * @exception CCAException with the following types: PortNotDefined, PortNotInUse.
 */
void
framework::Services_impl::releasePort (
  /* in */ const ::std::string& portName ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.releasePort)
  map<string, PortInstance*>::iterator iter = ports.find(portName);
  if(iter == ports.end()) {
    cerr << "Released an unknown port: " << portName << '\n';
    //throw CCAException("Released an unknown port: "+portName);
  }

  BabelPortInstance* pr = dynamic_cast<BabelPortInstance*> (iter->second);
  if (pr->porttype == BabelPortInstance::Provides) {
      cerr<<"Cannot call releasePort on a Provides port"<<endl;
      //throw CCAException("Cannot call releasePort on a Provides port");
  }

  if (!pr->decrementUseCount()) {
      cerr<<"Port released without correspond get"<<endl;
      //throw CCAException("Port released without correspond get");
  }
  // DO-NOT-DELETE splicer.end(framework.Services.releasePort)
}

/**
 * Creates a TypeMap, potentially to be used in subsequent
 * calls to describe a Port.  Initially, this map is empty.
 */
::gov::cca::TypeMap
framework::Services_impl::createTypeMap ()
throw ( 
  ::gov::cca::CCAException
)
{
  // DO-NOT-DELETE splicer.begin(framework.Services.createTypeMap)
  // insert implementation here
  return 0;
  // DO-NOT-DELETE splicer.end(framework.Services.createTypeMap)
}

/**
 * Register a request for a Port that will be retrieved subsequently 
 * with a call to getPort().
 * @param portName A string uniquely describing this port.  This string
 * must be unique for this component, over both uses and provides ports.
 * @param type A string desribing the type of this port.
 * @param properties A TypeMap describing optional properties
 * associated with this port. This can be a null pointer, which
 * indicates an empty list of properties.  Properties may be
 * obtained from createTypeMap or any other source.  The properties
 * be copied into the framework, and subsequent changes to the
 * properties object will have no effect on the properties
 * associated with this port.
 * In these properties, all frameworks recognize at least the
 * following keys and values in implementing registerUsesPort:
 * <pre xml:space="preserve">
 * key:              standard values (in string form)     default
 * "MAX_CONNECTIONS" any nonnegative integer, "unlimited".   1
 * "MIN_CONNECTIONS" any integer > 0.                        0
 * "ABLE_TO_PROXY"   "true", "false"                      "false"
 * </pre>
 * The component is not expected to work if the framework
 * has not satisfied the connection requirements.
 * The framework is allowed to return an error if it
 * is incapable of meeting the connection requirements,
 * e.g. it does not implement multiple uses ports.
 * The caller of registerUsesPort is not obligated to define
 * these properties. If left undefined, the default listed above is
 *       assumed.
 * @exception CCAException with the following types: PortAlreadyDefined, OutOfMemory.
 */
void
framework::Services_impl::registerUsesPort (
  /* in */ const ::std::string& portName,
  /* in */ const ::std::string& type,
  /* in */ ::gov::cca::TypeMap properties ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.registerUsesPort)
  map<string, PortInstance*>::iterator iter = ports.find(portName);
  if(iter != ports.end()){
    BabelPortInstance *pr=dynamic_cast<BabelPortInstance*> (iter->second);
    if(pr->porttype == BabelPortInstance::Provides)
      cerr<<"name conflict between uses and provides ports"<<endl;
      //throw CCAException("portName conflict between uses and provides ports");
    else {
      cerr << "registerUsesPort called twice portName = " << portName << ", portType = " << type << '\n';
      //throw CCAException("registerUsesPort called twice");
    }
  }
  ports.insert(make_pair(portName, new BabelPortInstance(portName, type, properties, BabelPortInstance::Uses)));


  // DO-NOT-DELETE splicer.end(framework.Services.registerUsesPort)
}

/**
 * Notify the framework that a Port, previously registered by this
 * component but currently not in use, is no longer desired. 
 * Unregistering a port that is currently 
 * in use (i.e. an unreleased getPort() being outstanding) 
 * is an error.
 * @param name The name of a registered Port.
 * @exception CCAException with the following types: UsesPortNotReleased, PortNotDefined.
 */
void
framework::Services_impl::unregisterUsesPort (
  /* in */ const ::std::string& portName ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.unregisterUsesPort)
   cerr << "unregisterUsesPort not done, portName=" << portName << '\n';
  // DO-NOT-DELETE splicer.end(framework.Services.unregisterUsesPort)
}

/**
 * Exposes a Port from this component to the framework.  
 * This Port is now available for the framework to connect 
 * to other components. 
 * @param inPort An abstract interface (tagged with CCA-ness
 * 	by inheriting from gov.cca.Port) the framework will
 * 	make available to other components.
 * 
 * @param portName string uniquely describing this port.  This string
 * must be unique for this component, over both uses and provides ports.
 * 
 * @param type string describing the type (class) of this port.
 * 
 * @param properties A TypeMap describing optional properties
 * associated with this port. This can be a null pointer, which
 * indicates an empty list of properties.  Properties may be
 * obtained from createTypeMap or any other source.  The properties
 * be copied into the framework, and subsequent changes to the
 * properties object will have no effect on the properties
 * associated with this port.
 * In these properties, all frameworks recognize at least the
 * following keys and values in implementing registerUsesPort:
 * <pre xml:space="preserve">
 * key:              standard values (in string form)     default
 * "MAX_CONNECTIONS" any nonnegative integer, "unlimited".   1
 * "MIN_CONNECTIONS" any integer > 0.                        0
 * "ABLE_TO_PROXY"   "true", "false"                      "false"
 * </pre>
 * The component is not expected to work if the framework
 * has not satisfied the connection requirements.
 * The framework is allowed to return an error if it
 * is incapable of meeting the connection requirements,
 * e.g. it does not implement multiple uses ports.
 * The caller of addProvidesPort is not obligated to define
 * these properties. If left undefined, the default listed above is
 * assumed.
 * @exception CCAException with the following types: PortAlreadyDefined, OutOfMemory.
 */
void
framework::Services_impl::addProvidesPort (
  /* in */ ::gov::cca::Port inPort,
  /* in */ const ::std::string& portName,
  /* in */ const ::std::string& type,
  /* in */ ::gov::cca::TypeMap properties ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.addProvidesPort)
  map<string, PortInstance*>::iterator iter = ports.find(portName);

  if (iter != ports.end()) {
    BabelPortInstance *pr = dynamic_cast<BabelPortInstance*> (iter->second);
    if (pr->porttype == BabelPortInstance::Uses) {
      cerr << "portName conflict between uses and provides ports" << endl;
    //throw CCAException("portName conflict between uses and provides ports");
    } else {
      cerr << "addProvidesPort called twice for " << portName << endl;
    //throw CCAException("addProvidesPort called twice");
    }
  }
  ports.insert(make_pair(portName, new BabelPortInstance(portName, type, properties, inPort, BabelPortInstance::Provides)));
  // DO-NOT-DELETE splicer.end(framework.Services.addProvidesPort)
}

/**
 * Returns the complete list of the properties for a Port.  This
 * includes the properties defined when the port was registered
 * (these properties can be modified by the framework), two special
 * properties "cca.portName" and "cca.portType", and any other
 * properties that the framework wishes to disclose to the component.
 * The framework may also choose to provide only the subset of input
 * properties (i.e. from addProvidesPort/registerUsesPort) that it
 * will honor.
 */
::gov::cca::TypeMap
framework::Services_impl::getPortProperties (
  /* in */ const ::std::string& name ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.getPortProperties)
  // insert implementation here
  return 0;
  // DO-NOT-DELETE splicer.end(framework.Services.getPortProperties)
}

/**
 * Notifies the framework that a previously exposed Port is no longer 
 * available for use. The Port being removed must exist
 * until this call returns, or a CCAException may occur.
 * @param name The name of a provided Port.
 * @exception PortNotDefined. In general, the framework will not dictate 
 * when the component chooses to stop offering services.
 */
void
framework::Services_impl::removeProvidesPort (
  /* in */ const ::std::string& portName ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.removeProvidesPort)
  cerr << "removeProvidesPort not done, portName=" << portName << '\n';
  // DO-NOT-DELETE splicer.end(framework.Services.removeProvidesPort)
}

/**
 * Get a reference to the component to which this 
 * Services object belongs. 
 */
::gov::cca::ComponentID
framework::Services_impl::getComponentID ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(framework.Services.getComponentID)
  // insert implementation here
  return 0;
  // DO-NOT-DELETE splicer.end(framework.Services.getComponentID)
}

/**
 * Obtain a callback for component destruction.
 * @param callback an object that implements the ComponentRelease
 * interface that will be called when the component is to be destroyed.
 * 
 * Register a callback to be executed when the component is going
 * to be destroyed.  During this callback, the Services object passed
 * through setServices will still be valid, but after all such
 * callbacks are made for a specific component, subsequent usage
 * of the Services object is not allowed/is undefined.
 */
void
framework::Services_impl::registerForRelease (
  /* in */ ::gov::cca::ComponentRelease callback ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(framework.Services.registerForRelease)
  // Insert-Code-Here {framework.Services.registerForRelease} (registerForRelease method)
  // DO-NOT-DELETE splicer.end(framework.Services.registerForRelease)
}


// DO-NOT-DELETE splicer.begin(framework.Services._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(framework.Services._misc)


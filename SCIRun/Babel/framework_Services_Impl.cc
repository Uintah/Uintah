// 
// File:          framework_Services_Impl.cc
// Symbol:        framework.Services-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20021108 00:42:45 EST
// Generated:     20021108 00:42:50 EST
// Description:   Server-side implementation for framework.Services
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 7
// source-url    = file:/.automount/linbox1/root/home/user2/sparker/SCIRun/cca/../src/SCIRun/Babel/framework.sidl
// 
#include "framework_Services_Impl.hh"

// DO-NOT-DELETE splicer.begin(framework.Services._includes)
// Put additional includes or other arbitrary code here...
#include <iostream.h>
// DO-NOT-DELETE splicer.end(framework.Services._includes)

// user defined constructor
void framework::Services_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(framework.Services._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(framework.Services._ctor)
}

// user defined destructor
void framework::Services_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(framework.Services._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(framework.Services._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
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
 * Ask for a previously registered Port; will return a Port or generate an error. 
 */
::govcca::Port
framework::Services_impl::getPort (
  /*in*/ const ::std::string& name ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.getPort)
  return getPortNonblocking(name);
  // DO-NOT-DELETE splicer.end(framework.Services.getPort)
}

/**
 * Ask for a previously registered Port and return that Port if it is
 * available or return null otherwise. 
 */
::govcca::Port
framework::Services_impl::getPortNonblocking (
  /*in*/ const ::std::string& name ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.getPortNonblocking)
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end())
    return 0;

  BabelPortInstance* pr = dynamic_cast<BabelPortInstance*> (iter->second);
  if(pr->porttype != BabelPortInstance::Uses)
    cerr<<"Cannot call getPort on a Provides port"<<endl;
    //throw CCAException("Cannot call getPort on a Provides port");

  pr->incrementUseCount();
  if(pr->connections.size() != 1)
    return 0;
  BabelPortInstance *pi=dynamic_cast<BabelPortInstance*> (pr->getPeer());
  return pi->port;
  // DO-NOT-DELETE splicer.end(framework.Services.getPortNonblocking)
}

/**
 * Modified according to Motion 31 
 */
void
framework::Services_impl::registerUsesPort (
  /*in*/ const ::std::string& name,
  /*in*/ const ::std::string& type,
  /*in*/ ::govcca::TypeMap properties ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.registerUsesPort)
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter != ports.end()){
    BabelPortInstance *pr=dynamic_cast<BabelPortInstance*> (iter->second);
    if(pr->porttype == BabelPortInstance::Provides)
      cerr<<"name conflict between uses and provides ports"<<endl;
      //throw CCAException("name conflict between uses and provides ports");
    else {
      cerr << "registerUsesPort called twice portName = " << name << ", portType = " << type << '\n';
      //throw CCAException("registerUsesPort called twice");
    }
  }
  ports.insert(make_pair(name, new BabelPortInstance(name, type, properties, BabelPortInstance::Uses)));


  // DO-NOT-DELETE splicer.end(framework.Services.registerUsesPort)
}

/**
 * Notify the framework that a Port, previously registered by this component,
 * is no longer needed. 
 */
void
framework::Services_impl::unregisterUsesPort (
  /*in*/ const ::std::string& name ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.unregisterUsesPort)
   cerr << "unregisterUsesPort not done, name=" << name << '\n';
  // DO-NOT-DELETE splicer.end(framework.Services.unregisterUsesPort)
}

/**
 * Exports a Port implemented by this component to the framework.  
 * This Port is now available for the framework to connect to other components. 
 * Modified according to Motion 31 
 */
void
framework::Services_impl::addProvidesPort (
  /*in*/ ::govcca::Port inPort,
  /*in*/ const ::std::string& name,
  /*in*/ const ::std::string& type,
  /*in*/ ::govcca::TypeMap properties ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.addProvidesPort)
  map<string, PortInstance*>::iterator iter = ports.find(name);

  if(iter != ports.end()){
    BabelPortInstance *pr=dynamic_cast<BabelPortInstance*> (iter->second);
    if(pr->porttype == BabelPortInstance::Provides)
      cerr<<"name conflict between uses and provides ports"<<endl;
    //throw CCAException("name conflict between uses and provides ports");
    else
      cerr<<"addProvidesPort called twice"<<endl;
    //throw CCAException("addProvidesPort called twice");
  }
  ports.insert(make_pair(name, new BabelPortInstance(name, type, properties, inPort, BabelPortInstance::Provides)));
  // DO-NOT-DELETE splicer.end(framework.Services.addProvidesPort)
}

/**
 * Notifies the framework that a previously exported Port is no longer 
 * available for use.
 */
void
framework::Services_impl::removeProvidesPort (
  /*in*/ const ::std::string& name ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.removeProvidesPort)
  cerr << "removeProvidesPort not done, name=" << name << '\n';
  // DO-NOT-DELETE splicer.end(framework.Services.removeProvidesPort)
}

/**
 * Notifies the framework that this component is finished with this Port.   
 * releasePort() method calls exactly match getPort() mehtod calls.  After 
 * releasePort() is invoked all references to the released Port become invalid. 
 */
void
framework::Services_impl::releasePort (
  /*in*/ const ::std::string& name ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.releasePort)
  map<string, PortInstance*>::iterator iter = ports.find(name);
  if(iter == ports.end()){
    cerr << "Released an unknown port: " << name << '\n';
    //throw CCAException("Released an unknown port: "+name);
  }

  BabelPortInstance* pr = dynamic_cast<BabelPortInstance*> (iter->second);
  if(pr->porttype != BabelPortInstance::Uses)
    cerr<<"Cannot call releasePort on a Provides port"<<endl;
  //throw CCAException("Cannot call releasePort on a Provides port");

  if(!pr->decrementUseCount())
    cerr<<"Port released without correspond get"<<endl;
    //throw CCAException("Port released without correspond get");
  // DO-NOT-DELETE splicer.end(framework.Services.releasePort)
}

/**
 * Method:  createTypeMap[]
 */
::govcca::TypeMap
framework::Services_impl::createTypeMap () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(framework.Services.createTypeMap)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.Services.createTypeMap)
}

/**
 * Method:  getPortProperties[]
 */
::govcca::TypeMap
framework::Services_impl::getPortProperties (
  /*in*/ const ::std::string& portName ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(framework.Services.getPortProperties)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.Services.getPortProperties)
}

/**
 * Get a reference to the component to which this Services object belongs. 
 */
::govcca::ComponentID
framework::Services_impl::getComponentID () 
throw () 

{
  // DO-NOT-DELETE splicer.begin(framework.Services.getComponentID)
  // insert implementation here
  // DO-NOT-DELETE splicer.end(framework.Services.getComponentID)
}


// DO-NOT-DELETE splicer.begin(framework.Services._misc)
// Put miscellaneous code here
// DO-NOT-DELETE splicer.end(framework.Services._misc)


//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : spectest.cc
//    Author : Martin Cole
//    Date   : Tue Aug 21 11:17:14 2001

#include <testprograms/Component/spectest/spectest_impl.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>

#include <iostream>
#include <algorithm>
#include <sstream>
#include <unistd.h>
#include <stdlib.h>
using namespace std;

//using spectest::ref;
//using spectest::Server;

//using SSIDL::array1;
//using std::istringstream;
//using std::ostringstream;

using namespace SCIRun;

namespace spectest {

Port_impl::Port_impl()
{
  cerr << "Port_impl constructor" << endl;
}

Port_impl::~Port_impl()
{
  cerr << "Port_impl destructor" << endl;
}


PortInfo_impl::PortInfo_impl()
{
  cerr << "PortInfo_impl constructor" << endl;
}

PortInfo_impl::~PortInfo_impl()
{
  cerr << "PortInfo_impl destructor" << endl;
}

::SSIDL::string 
PortInfo_impl::getType()
{
  cerr << "PortInfo_impl::getType()" << endl;
  return type_;
}
::SSIDL::string 
PortInfo_impl::getName()
{
  cerr << "PortInfo_impl::getName()" << endl;
  return name_;
}
::SSIDL::string 
PortInfo_impl::getProperty(const ::SSIDL::string& /*name*/)
{
  cerr << "PortInfo_impl::getProperty(const ::SSIDL::string& name)" << endl;

//    map<string, string>::iterator iter = properties_.find(name);
//    if (iter != properties_.end()) {
//      return (*iter).second;
//    }

  return "no such property";
}

Services_impl::Services_impl()
{
  cerr << "Services_impl constructor" << endl;
}

Services_impl::~Services_impl()
{
  cerr << "Services_impl destructor" << endl;
}

Port::pointer
Services_impl::getPort(const ::SSIDL::string& /*name*/)
{
  Port::pointer p;
  cerr << "Services_impl::getPort(const ::SSIDL::string& name)" << endl;
  return p;
}
Port::pointer
Services_impl::getPortNonblocking(const ::SSIDL::string& /*name*/)
{
  Port::pointer p;
  cerr << "Services_impl::getPortNonblocking(const ::SSIDL::string& name)" 
       << endl;
  return p;
}
PortInfo::pointer
Services_impl::createPortInfo(const ::SSIDL::string& name, 
			  const ::SSIDL::string& type, 
			  const ::SSIDL::array1< ::SSIDL::string> &properties )
{
  PortInfo_impl *pi = new PortInfo_impl;
  pi->setName(name);
  pi->setType(type);
  pi->setProperties(properties);
  cerr << "Services_impl::createPortInfo(const ::SSIDL::string& name, " << endl;
  return PortInfo::pointer(pi);
}
void 
Services_impl::registerUsesPort(const PortInfo::pointer &/*name_and_type*/)
{
  cerr << "Services_impl::registerUsesPort(const PortInfo &name_and_type)" 
       << endl;
}
void 
Services_impl::unregisterUsesPort(const ::SSIDL::string& /*name*/)
{
  cerr << "Services_impl::unregisterUsesPort(const ::SSIDL::string&name)" 
       << endl;
}
void 
Services_impl::addProvidesPort(const Port::pointer &inPort, const PortInfo::pointer &name)
{
  cerr << "Services_impl::addProvidesPort" << endl;
  // store off the port.  make sure it does not already exist.
  port_map_t::iterator iter = provides_.find(name->getName());
  if (iter == provides_.end()) {
    provides_[name->getName()] = PortData(inPort, name);
  } else {
    // trying to add port with a name that already exists.
    //throw DuplicatePortException;
    cerr << "throw DuplicatePortException;" << endl;
  }
}
void 
Services_impl::removeProvidesPort(const ::SSIDL::string&name)
{
  cerr << "Services_impl::removeProvidesPort(const ::SSIDL::string&name)" 
       << endl;

  // store off the port.  make sure it does not already exist.
  port_map_t::iterator iter = provides_.find(name);
  if (iter == provides_.end()) {
    provides_.erase(iter);
  }
}
void 
Services_impl::releasePort(const ::SSIDL::string& /*name*/)
{
  cerr << "Services_impl::releasePort(const ::SSIDL::string&name)" << endl;
}
ComponentID ::pointer
Services_impl::getComponentID()
{
  ComponentID::pointer cid;
  cerr << "Services_impl::getComponentID()" << endl;
  return cid;
}

Component_impl::Component_impl()
{
  cerr << "Component_impl constructor" << endl;
}

Component_impl::~Component_impl()
{
  cerr << "Component_impl destructor" << endl;
}

void 
Component_impl::setServices(const Services::pointer &svc)
{
  cerr << "Component_impl::setServices(const Services &svc)" << endl;
  services_ = svc;
}

ComponentID_impl::ComponentID_impl()
{
  cerr << "ComponentID_impl constructor" << endl;
}

ComponentID_impl::~ComponentID_impl()
{
  cerr << "ComponentID_impl destructor" << endl;
}

::SSIDL::string 
ComponentID_impl::toString()
{
  cerr << "ComponentID_impl::toString()" << endl;
  return "bogus";
}

GoPort_impl::GoPort_impl()
{
  cerr << "GoPort_impl constructor" << endl;
}

GoPort_impl::~GoPort_impl()
{
  cerr << "GoPort_impl destructor" << endl;
}

ConnectionEventService_impl::ConnectionEventService_impl()
{
  cerr << "ConnectionEventService_impl constructor" << endl;
}

ConnectionEventService_impl::~ConnectionEventService_impl()
{
  cerr << "ConnectionEventService_impl destructor" << endl;
}

void 
ConnectionEventService_impl::
addConnectionEventListener(int /*connectionEventType*/, 
			   const ConnectionEventListener::pointer &/*l*/)
{
  cerr << "addConnectionEventListener(int connectionEventType, " << endl;
}

void 
ConnectionEventService_impl::
removeConnectionEventListener(int /*connectionEventType*/, 
			      const ConnectionEventListener::pointer &/*l*/)
{
  cerr << "removeConnectionEventListener(int connectionEventType, " << endl;
}

ConnectionEventListener_impl::ConnectionEventListener_impl()
{
  cerr << "ConnectionEventListener_impl constructor" << endl;
}

ConnectionEventListener_impl::~ConnectionEventListener_impl()
{
  cerr << "ConnectionEventListener_impl destructor" << endl;
}

void 
ConnectionEventListener_impl::connectionActivity(const ConnectionEvent::pointer &/*evt*/)
{
  cerr << "ConnectionEventListener_impl::connectionActivity" << endl;
}

ConnectionEvent_impl::ConnectionEvent_impl()
{
  cerr << "ConnectionEvent_impl constructor" << endl;
}

ConnectionEvent_impl::~ConnectionEvent_impl()
{
  cerr << "ConnectionEvent_impl destructor" << endl;
}

int 
ConnectionEvent_impl::getEventType()
{
  cerr << "ConnectionEvent_impl::getEventType()" << endl;
  return Error;
}

PortInfo::pointer
ConnectionEvent_impl::getPortInfo()
{
  PortInfo::pointer pi;
  cerr << "ConnectionEvent_impl::getPortInfo()" << endl;
  return pi;
}

/* ---------------------------------------------------------------------
 * Below is all of the non specification test code.
 * --------------------------------------------------------------------*/

IntegerStream_impl::IntegerStream_impl()
{cerr << "IntegerStream_impl::ctor()" << endl;}
IntegerStream_impl::~IntegerStream_impl()
{cerr << "IntegerStream_impl::dtor()" << endl;}

void
IntegerStream_impl::push(int i) 
{
  cerr << "IntegerStream_impl::push(int i)" << endl;
  stream_.push(i);
}

int
IntegerStream_impl::pop() 
{
  cerr << "IntegerStream_impl::pop()" << endl;
  int rval = stream_.front();
  stream_.pop();
  return rval;
}

bool
IntegerStream_impl::is_full() 
{
  cerr << "IntegerStream_impl::is_full()" << endl;
  return stream_.size() >= 100;
}

bool
IntegerStream_impl::is_empty() 
{
  cerr << "IntegerStream_impl::is_empty()" << endl;
  return stream_.size() == 0;
}


RandomInt::RandomInt() :
  istr_(new IntegerStream_impl)
{cerr << "RandomInt::ctor()" << endl;}
RandomInt::~RandomInt()
{cerr << "RandomInt::dtor()" << endl;}

void
RandomInt::go()
{
  srand(69);

  for(;;) {

    if (istr_->is_full()) {
      sleep(1);
    } else {
      istr_->push(rand());
    }
  }
}

void 
RandomInt::setServices(const CCA::Services::pointer &svc)
{
  cout << "RandomInt::setServices(const CCA::Services &svc)" << endl; 
  Component_impl::setServices(svc);

  ::SSIDL::array1< ::SSIDL::string> prop;
  prop.push_back("property one");
  PortInfo::pointer pi = services_->createPortInfo("RandomIntStream", 
					  "IntegerStream",
					  prop);
  services_->addProvidesPort(istr_, pi);
}

ConsumerInt::ConsumerInt() :
  istr_(new IntegerStream_impl)
{cerr << "ConsumerInt::ctor()" << endl;}
ConsumerInt::~ConsumerInt()
{cerr << "ConsumerInt::dtor()" << endl;}

void
ConsumerInt::go()
{
  for(;;) {

    if (istr_->is_empty()) {
      sleep(1);
    } else {
      cerr << "ConsumerInt: " << istr_->pop();
    }
  }
}

void 
ConsumerInt::setServices(const CCA::Services::pointer &svc)
{
  cout << "ConsumerInt::setServices(const CCA::Services &svc)" << endl; 
  Component_impl::setServices(svc);

  ::SSIDL::array1< ::SSIDL::string> prop;
  prop.push_back("property one");
  PortInfo::pointer pi = services_->createPortInfo("RandomIntStream", 
					  "IntegerStream",
					  prop);

  services_->registerUsesPort(pi);
}

Framework_impl::Framework_impl() :
  services_(new Services_impl)
{}
Framework_impl::~Framework_impl()
{}

CCA::Services::pointer
Framework_impl::get_services() 
{
  cerr << "Framework_impl::get_services()" << endl;
  return services_;
}



} // end spectest namespace.

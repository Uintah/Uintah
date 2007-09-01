// 
// File:          HelloClient_Component_Impl.cxx
// Symbol:        HelloClient.Component-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for HelloClient.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "HelloClient_Component_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
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
// DO-NOT-DELETE splicer.begin(HelloClient.Component._includes)
// Insert-Code-Here {HelloClient.Component._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(HelloClient.Component._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
HelloClient::Component_impl::Component_impl() : StubBase(reinterpret_cast< 
  void*>(::HelloClient::Component::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(HelloClient.Component._ctor2)
  // Insert-Code-Here {HelloClient.Component._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(HelloClient.Component._ctor2)
}

// user defined constructor
void HelloClient::Component_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(HelloClient.Component._ctor)
  // Insert-Code-Here {HelloClient.Component._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(HelloClient.Component._ctor)
}

// user defined destructor
void HelloClient::Component_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(HelloClient.Component._dtor)
  // Insert-Code-Here {HelloClient.Component._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(HelloClient.Component._dtor)
}

// static class initializer
void HelloClient::Component_impl::_load() {
  // DO-NOT-DELETE splicer.begin(HelloClient.Component._load)
  // Insert-Code-Here {HelloClient.Component._load} (class initialization)
  // DO-NOT-DELETE splicer.end(HelloClient.Component._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * The following method starts the component.
 */
int32_t
HelloClient::Component_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(HelloClient.Component.go)
  gov::cca::Port p = svc.getPort("msgport-up");
  HelloServer::MsgPort s = ::sidl::babel_cast<HelloServer::MsgPort>(p);
  if(s._is_nil()) {
    std::cerr << "getPort() returns null" << ::std::endl;
    return -1;
  }
  s.printMsg("This works\n");
  return 0;
  // DO-NOT-DELETE splicer.end(HelloClient.Component.go)
}

/**
 * Method <code>setServices</code> is called by the framework.
 */
void
HelloClient::Component_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(HelloClient.Component.setServices)
  svc = services;
  services.registerUsesPort("msgport-up", "HelloServer.MsgPort", 0);
  services.addProvidesPort(*this, "goport", "gov.cca.ports.GoPort", 0);
  // DO-NOT-DELETE splicer.end(HelloClient.Component.setServices)
}


// DO-NOT-DELETE splicer.begin(HelloClient.Component._misc)
// Insert-Code-Here {HelloClient.Component._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(HelloClient.Component._misc)


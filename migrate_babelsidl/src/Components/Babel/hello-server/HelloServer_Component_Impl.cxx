// 
// File:          HelloServer_Component_Impl.cxx
// Symbol:        HelloServer.Component-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for HelloServer.Component
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "HelloServer_Component_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(HelloServer.Component._includes)
#include <HelloServer_MsgPort.hxx>
// DO-NOT-DELETE splicer.end(HelloServer.Component._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
HelloServer::Component_impl::Component_impl() : StubBase(reinterpret_cast< 
  void*>(::HelloServer::Component::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(HelloServer.Component._ctor2)
  // Insert-Code-Here {HelloServer.Component._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(HelloServer.Component._ctor2)
}

// user defined constructor
void HelloServer::Component_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(HelloServer.Component._ctor)
  // Insert-Code-Here {HelloServer.Component._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(HelloServer.Component._ctor)
}

// user defined destructor
void HelloServer::Component_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(HelloServer.Component._dtor)
  // Insert-Code-Here {HelloServer.Component._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(HelloServer.Component._dtor)
}

// static class initializer
void HelloServer::Component_impl::_load() {
  // DO-NOT-DELETE splicer.begin(HelloServer.Component._load)
  // Insert-Code-Here {HelloServer.Component._load} (class initialization)
  // DO-NOT-DELETE splicer.end(HelloServer.Component._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method <code>setServices</code> is called by the framework.
 */
void
HelloServer::Component_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(HelloServer.Component.setServices)
  HelloServer::MsgPort msgp = ::HelloServer::MsgPort::_create();

  ::gov::cca::Port msgPort = ::sidl::babel_cast<gov::cca::Port>(msgp);
  if (msgPort._not_nil()) {
    services.addProvidesPort(msgPort, "msgport-pp", "HelloServer.MsgPort", 0);
  }
  // DO-NOT-DELETE splicer.end(HelloServer.Component.setServices)
}


// DO-NOT-DELETE splicer.begin(HelloServer.Component._misc)
// Insert-Code-Here {HelloServer.Component._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(HelloServer.Component._misc)


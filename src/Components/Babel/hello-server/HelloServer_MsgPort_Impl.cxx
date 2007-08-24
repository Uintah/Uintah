// 
// File:          HelloServer_MsgPort_Impl.cxx
// Symbol:        HelloServer.MsgPort-v1.0
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for HelloServer.MsgPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "HelloServer_MsgPort_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._includes)
#include <iostream>
// DO-NOT-DELETE splicer.end(HelloServer.MsgPort._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
HelloServer::MsgPort_impl::MsgPort_impl() : StubBase(reinterpret_cast< void*>(
  ::HelloServer::MsgPort::_wrapObj(reinterpret_cast< void*>(this))),false) , 
  _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._ctor2)
  // Insert-Code-Here {HelloServer.MsgPort._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(HelloServer.MsgPort._ctor2)
}

// user defined constructor
void HelloServer::MsgPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._ctor)
  // Insert-Code-Here {HelloServer.MsgPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(HelloServer.MsgPort._ctor)
}

// user defined destructor
void HelloServer::MsgPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._dtor)
  // Insert-Code-Here {HelloServer.MsgPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(HelloServer.MsgPort._dtor)
}

// static class initializer
void HelloServer::MsgPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._load)
  // Insert-Code-Here {HelloServer.MsgPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(HelloServer.MsgPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  printMsg[]
 */
void
HelloServer::MsgPort_impl::printMsg_impl (
  /* in */const ::std::string& msg ) 
{
  // DO-NOT-DELETE splicer.begin(HelloServer.MsgPort.printMsg)
  std::cerr << "Received message: " << msg << "\n";
  // DO-NOT-DELETE splicer.end(HelloServer.MsgPort.printMsg)
}


// DO-NOT-DELETE splicer.begin(HelloServer.MsgPort._misc)
// Insert-Code-Here {HelloServer.MsgPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(HelloServer.MsgPort._misc)


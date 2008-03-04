// 
// File:          NewPort_StringPort_Impl.cxx
// Symbol:        NewPort.StringPort-v1.0
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for NewPort.StringPort
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "NewPort_StringPort_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(NewPort.StringPort._includes)
// Insert-Code-Here {NewPort.StringPort._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(NewPort.StringPort._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
NewPort::StringPort_impl::StringPort_impl() : StubBase(reinterpret_cast< void*>(
  ::NewPort::StringPort::_wrapObj(reinterpret_cast< void*>(this))),false) , 
  _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._ctor2)
  // Insert-Code-Here {NewPort.StringPort._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._ctor2)
}

// user defined constructor
void NewPort::StringPort_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._ctor)
  // Insert-Code-Here {NewPort.StringPort._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._ctor)
}

// user defined destructor
void NewPort::StringPort_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._dtor)
  // Insert-Code-Here {NewPort.StringPort._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._dtor)
}

// static class initializer
void NewPort::StringPort_impl::_load() {
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort._load)
  // Insert-Code-Here {NewPort.StringPort._load} (class initialization)
  // DO-NOT-DELETE splicer.end(NewPort.StringPort._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  getString[]
 */
::std::string
NewPort::StringPort_impl::getString_impl () 

{
  // DO-NOT-DELETE splicer.begin(NewPort.StringPort.getString)
  // Insert-Code-Here {NewPort.StringPort.getString} (getString method)
  return "NewPort::StringPort_impl::getString CALLED";
  // DO-NOT-DELETE splicer.end(NewPort.StringPort.getString)
}


// DO-NOT-DELETE splicer.begin(NewPort.StringPort._misc)
// Insert-Code-Here {NewPort.StringPort._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(NewPort.StringPort._misc)


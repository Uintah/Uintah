// 
// File:          scijump_Event_Impl.cxx
// Symbol:        scijump.Event-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.Event
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_Event_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_io_Deserializer_hxx
#include "sidl_io_Deserializer.hxx"
#endif
#ifndef included_sidl_io_Serializer_hxx
#include "sidl_io_Serializer.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(scijump.Event._includes)
// Insert-Code-Here {scijump.Event._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.Event._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::Event_impl::Event_impl() : StubBase(reinterpret_cast< void*>(
  ::scijump::Event::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(scijump.Event._ctor2)
  // Insert-Code-Here {scijump.Event._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.Event._ctor2)
}

// user defined constructor
void scijump::Event_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.Event._ctor)
  // Insert-Code-Here {scijump.Event._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.Event._ctor)
}

// user defined destructor
void scijump::Event_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.Event._dtor)
  // Insert-Code-Here {scijump.Event._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.Event._dtor)
}

// static class initializer
void scijump::Event_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.Event._load)
  // Insert-Code-Here {scijump.Event._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.Event._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  setHeader[]
 */
void
scijump::Event_impl::setHeader_impl (
  /* in */::gov::cca::TypeMap& h ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Event.setHeader)
  header = h;
  // DO-NOT-DELETE splicer.end(scijump.Event.setHeader)
}

/**
 * Method:  setBody[]
 */
void
scijump::Event_impl::setBody_impl (
  /* in */::gov::cca::TypeMap& b ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Event.setBody)
  body = b;
  // DO-NOT-DELETE splicer.end(scijump.Event.setBody)
}

/**
 *  Return the event's header. The header is usually generated
 * by the framework and holds bookkeeping information
 */
::gov::cca::TypeMap
scijump::Event_impl::getHeader_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.Event.getHeader)
  return header;
  // DO-NOT-DELETE splicer.end(scijump.Event.getHeader)
}

/**
 *  Returs the event's body. The body is the information the
 * publisher is sending to the subscribers
 */
::gov::cca::TypeMap
scijump::Event_impl::getBody_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.Event.getBody)
  return body;
  // DO-NOT-DELETE splicer.end(scijump.Event.getBody)
}

/**
 * Method:  packObj[]
 */
void
scijump::Event_impl::packObj_impl (
  /* in */::sidl::io::Serializer& ser ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Event.packObj)
  // Insert-Code-Here {scijump.Event.packObj} (packObj method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.Event.packObj)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "packObj");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.Event.packObj)
  // DO-NOT-DELETE splicer.end(scijump.Event.packObj)
}

/**
 * Method:  unpackObj[]
 */
void
scijump::Event_impl::unpackObj_impl (
  /* in */::sidl::io::Deserializer& des ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Event.unpackObj)
  // Insert-Code-Here {scijump.Event.unpackObj} (unpackObj method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.Event.unpackObj)
  ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
  ex.setNote("This method has not been implemented");
  ex.add(__FILE__, __LINE__, "unpackObj");
  throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.Event.unpackObj)
  // DO-NOT-DELETE splicer.end(scijump.Event.unpackObj)
}


// DO-NOT-DELETE splicer.begin(scijump.Event._misc)
// Insert-Code-Here {scijump.Event._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.Event._misc)


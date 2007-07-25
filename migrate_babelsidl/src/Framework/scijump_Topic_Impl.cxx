// 
// File:          scijump_Topic_Impl.cxx
// Symbol:        scijump.Topic-v0.2.1
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for scijump.Topic
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_Topic_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_TypeMap_hxx
#include "gov_cca_TypeMap.hxx"
#endif
#ifndef included_sci_cca_EventServiceException_hxx
#include "sci_cca_EventServiceException.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.Topic._includes)
// Insert-Code-Here {scijump.Topic._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(scijump.Topic._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::Topic_impl::Topic_impl() : StubBase(reinterpret_cast< void*>(
  ::scijump::Topic::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
  true){ 
  // DO-NOT-DELETE splicer.begin(scijump.Topic._ctor2)
  // Insert-Code-Here {scijump.Topic._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.Topic._ctor2)
}

// user defined constructor
void scijump::Topic_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.Topic._ctor)
  // Insert-Code-Here {scijump.Topic._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.Topic._ctor)
}

// user defined destructor
void scijump::Topic_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.Topic._dtor)
  // Insert-Code-Here {scijump.Topic._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.Topic._dtor)
}

// static class initializer
void scijump::Topic_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.Topic._load)
  // Insert-Code-Here {scijump.Topic._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.Topic._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * @return Topic name (from EventService.createTopic)
 */
::std::string
scijump::Topic_impl::getTopicName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.Topic.getTopicName)
  return topicName;
  // DO-NOT-DELETE splicer.end(scijump.Topic.getTopicName)
}

/**
 *  Publish an event. 
 */
void
scijump::Topic_impl::sendEvent_impl (
  /* in */const ::std::string& name,
  /* in */::gov::cca::TypeMap eventBody ) 
// throws:
//     ::sci::cca::EventServiceException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.Topic.sendEvent)
  // Insert-Code-Here {scijump.Topic.sendEvent} (sendEvent method)
  // 
  // This method has not been implemented
  // 
    ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
    ex.setNote("This method has not been implemented");
    ex.add(__FILE__, __LINE__, "sendEvent");
    throw ex;
  // DO-NOT-DELETE splicer.end(scijump.Topic.sendEvent)
}


// DO-NOT-DELETE splicer.begin(scijump.Topic._misc)
// Insert-Code-Here {scijump.Topic._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.Topic._misc)


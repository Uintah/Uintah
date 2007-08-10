// 
// File:          scijump_Subscription_Impl.cxx
// Symbol:        scijump.Subscription-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.Subscription
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_Subscription_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_Event_hxx
#include "sci_cca_Event.hxx"
#endif
#ifndef included_sci_cca_EventListener_hxx
#include "sci_cca_EventListener.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.Subscription._includes)
#include <scijump_EventServiceException.hxx>
// DO-NOT-DELETE splicer.end(scijump.Subscription._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::Subscription_impl::Subscription_impl() : StubBase(reinterpret_cast< 
  void*>(::scijump::Subscription::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.Subscription._ctor2)
  // Insert-Code-Here {scijump.Subscription._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.Subscription._ctor2)
}

// user defined constructor
void scijump::Subscription_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.Subscription._ctor)
  // Insert-Code-Here {scijump.Subscription._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.Subscription._ctor)
}

// user defined destructor
void scijump::Subscription_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.Subscription._dtor)
  // Insert-Code-Here {scijump.Subscription._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.Subscription._dtor)
}

// static class initializer
void scijump::Subscription_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.Subscription._load)
  // Insert-Code-Here {scijump.Subscription._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.Subscription._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::Subscription_impl::initialize_impl (
  /* in */const ::std::string& subscriptionName,
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Subscription.initialize)
  this->subscriptionName = subscriptionName;
  this->framework = framework;
  // DO-NOT-DELETE splicer.end(scijump.Subscription.initialize)
}

/**
 * Method:  processEvents[]
 */
void
scijump::Subscription_impl::processEvents_impl (
  /* in array<sci.cca.Event> */::sidl::array< ::sci::cca::Event>& eventList ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Subscription.processEvents)
  if (eventListenerMap.empty()) {
    return;
  }

  if (eventList.length() == 0) {
    return;
  }

  for (EventListenerMap::iterator eventListenerIter = eventListenerMap.begin();
       eventListenerIter != eventListenerMap.end(); eventListenerIter++) {

    // Call processEvent() on each Listener
    for (unsigned int i = 0; i < eventList.length(); i++) {
      // Call processEvent() on each event
      (eventListenerIter->second).processEvent(subscriptionName, eventList.get(i));
    }
  }
  // DO-NOT-DELETE splicer.end(scijump.Subscription.processEvents)
}

/**
 *  Adds a listener to the collection of listeners for this Subscription.
 * 
 * @listenerKey - It is used as an index to the collection (STL
 * map) and the parameter \em theListener is a
 * pointer to the /em Listener class.
 * @theListener - A pointer to the object that will listen for events.
 */
void
scijump::Subscription_impl::registerEventListener_impl (
  /* in */const ::std::string& listenerKey,
  /* in */::sci::cca::EventListener& theListener ) 
// throws:
//     ::sci::cca::EventServiceException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.Subscription.registerEventListener)
  if (listenerKey.empty()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Listener key is empty");
    throw ex; 
  }

  if (theListener._is_nil()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Listener pointer is nil");
    throw ex; 
  }
  EventListenerMap::iterator iter =  eventListenerMap.find(listenerKey);
  if (iter != eventListenerMap.end()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Listener key already present");
    throw ex; 
  }

  eventListenerMap[listenerKey] = theListener;
  // DO-NOT-DELETE splicer.end(scijump.Subscription.registerEventListener)
}

/**
 * Removes a listener from the collection of listeners for this Topic.
 * 
 * @listenerKey - It is used as an index to remove this listener.
 */
void
scijump::Subscription_impl::unregisterEventListener_impl (
  /* in */const ::std::string& listenerKey ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Subscription.unregisterEventListener)
  if (listenerKey.empty()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Listener key is empty");
    throw ex; 
  }

  EventListenerMap::iterator iter =  eventListenerMap.find(listenerKey);
  if (iter == eventListenerMap.end()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Listener key is not registered");
    throw ex; 
  }

  eventListenerMap.erase(iter);
  // DO-NOT-DELETE splicer.end(scijump.Subscription.unregisterEventListener)
}

/**
 *  Returns the name for this Subscription object 
 */
::std::string
scijump::Subscription_impl::getSubscriptionName_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.Subscription.getSubscriptionName)
  return subscriptionName;
  // DO-NOT-DELETE splicer.end(scijump.Subscription.getSubscriptionName)
}


// DO-NOT-DELETE splicer.begin(scijump.Subscription._misc)
// Insert-Code-Here {scijump.Subscription._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.Subscription._misc)


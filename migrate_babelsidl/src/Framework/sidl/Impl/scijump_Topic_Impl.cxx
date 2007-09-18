// 
// File:          scijump_Topic_Impl.cxx
// Symbol:        scijump.Topic-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
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
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_EventServiceException_hxx
#include "sci_cca_EventServiceException.hxx"
#endif
#ifndef included_scijump_Subscription_hxx
#include "scijump_Subscription.hxx"
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
#include <scijump_EventServiceException.hxx>
// DO-NOT-DELETE splicer.end(scijump.Topic._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
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
 * Method:  initialize[]
 */
void
scijump::Topic_impl::initialize_impl (
  /* in */const ::std::string& topicName,
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Topic.initialize)
  this->topicName = topicName;
  this->framework = framework;
  // DO-NOT-DELETE splicer.end(scijump.Topic.initialize)
}

/**
 * Method:  addSubscription[]
 */
void
scijump::Topic_impl::addSubscription_impl (
  /* in */const ::std::string& topicName,
  /* in */::scijump::Subscription& theSubscription ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Topic.addSubscription)
  if (topicName.empty()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Topic name is empty");
    throw ex;
  }

  if (theSubscription._is_nil()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Subscription is null");
    throw ex;
  }

  SubscriptionMap::iterator iter =  subscriptionMap.find(topicName);
  if (iter != subscriptionMap.end()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Subscription already registered for this topic");
    throw ex;   
  }
  subscriptionMap[topicName] = theSubscription;
  // DO-NOT-DELETE splicer.end(scijump.Topic.addSubscription)
}

/**
 * Method:  removeSubscription[]
 */
void
scijump::Topic_impl::removeSubscription_impl (
  /* in */const ::std::string& topicName ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.Topic.removeSubscription)
  if (topicName.empty()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Topic name is empty");
    throw ex;
  }

  SubscriptionMap::iterator iter =  subscriptionMap.find(topicName);
  if (iter == subscriptionMap.end()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Subscription not registered for this topic");
    throw ex;   
  }
  subscriptionMap.erase(iter);
  // DO-NOT-DELETE splicer.end(scijump.Topic.removeSubscription)
}

/**
 * Method:  processEvents[]
 */
void
scijump::Topic_impl::processEvents_impl () 

{
  // DO-NOT-DELETE splicer.begin(scijump.Topic.processEvents)
  sidl::array< sci::cca::Event> eventArr = 
    sidl::array< sci::cca::Event>::create1d(eventList.size());
  for(int i=0; i < eventList.size(); i++) {
    eventArr.set(i,eventList[i]);
  }

  for (SubscriptionMap::iterator subscriptionIter = subscriptionMap.begin();
       subscriptionIter != subscriptionMap.end(); subscriptionIter++) {

    Subscription subscriptionPtr = subscriptionIter->second;
    if (subscriptionPtr._is_nil()) {
      EventServiceException ex = EventServiceException::_create();
      ex.setNote("Subscription is nil");
      throw ex;         
    }
    subscriptionPtr.processEvents(eventArr);
  }

  eventList.clear();
  // DO-NOT-DELETE splicer.end(scijump.Topic.processEvents)
}

/**
 *  Returns the topic name associated with this object 
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
 * 
 * @eventName - The name of this event. It is perhaps not a crucial
 * piece of information. Can be inserted into the
 * header or the body of the event by the event
 * service.
 * @eventBody - A typemap containing all the information to be
 * sent out.
 */
void
scijump::Topic_impl::sendEvent_impl (
  /* in */const ::std::string& eventName,
  /* in */::gov::cca::TypeMap& eventBody ) 
// throws:
//     ::sci::cca::EventServiceException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.Topic.sendEvent)
  scijump::Event event = scijump::Event::_create();
  gov::cca::TypeMap eventHeader = framework.createTypeMap();
  eventHeader.putString("eventName",eventName);

  //TODO: framework should put other info in event header
  event.setHeader(eventHeader);
  event.setBody(eventBody);
  
  eventList.push_back(event);
  // DO-NOT-DELETE splicer.end(scijump.Topic.sendEvent)
}


// DO-NOT-DELETE splicer.begin(scijump.Topic._misc)
// Insert-Code-Here {scijump.Topic._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(scijump.Topic._misc)


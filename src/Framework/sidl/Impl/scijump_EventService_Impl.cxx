// 
// File:          scijump_EventService_Impl.cxx
// Symbol:        scijump.EventService-v0.2.1
// Symbol Type:   class
// Babel Version: 1.1.0
// Description:   Server-side implementation for scijump.EventService
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "scijump_EventService_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_EventServiceException_hxx
#include "sci_cca_EventServiceException.hxx"
#endif
#ifndef included_sci_cca_Subscription_hxx
#include "sci_cca_Subscription.hxx"
#endif
#ifndef included_sci_cca_Topic_hxx
#include "sci_cca_Topic.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
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
// DO-NOT-DELETE splicer.begin(scijump.EventService._includes)
#include <scijump_EventServiceException.hxx>
// DO-NOT-DELETE splicer.end(scijump.EventService._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
scijump::EventService_impl::EventService_impl() : StubBase(reinterpret_cast< 
  void*>(::scijump::EventService::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(scijump.EventService._ctor2)
  // Insert-Code-Here {scijump.EventService._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(scijump.EventService._ctor2)
}

// user defined constructor
void scijump::EventService_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(scijump.EventService._ctor)
  // Insert-Code-Here {scijump.EventService._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(scijump.EventService._ctor)
}

// user defined destructor
void scijump::EventService_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(scijump.EventService._dtor)
  // Insert-Code-Here {scijump.EventService._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(scijump.EventService._dtor)
}

// static class initializer
void scijump::EventService_impl::_load() {
  // DO-NOT-DELETE splicer.begin(scijump.EventService._load)
  // Insert-Code-Here {scijump.EventService._load} (class initialization)
  // DO-NOT-DELETE splicer.end(scijump.EventService._load)
}

// user defined static methods:
/**
 * Method:  create[]
 */
::sci::cca::core::FrameworkService
scijump::EventService_impl::create_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.EventService.create)
  // Insert-Code-Here {scijump.EventService.create} (create method)
  // 
  // This method has not been implemented
  // 
  // DO-DELETE-WHEN-IMPLEMENTING exception.begin(scijump.EventService.create)
//   ::sidl::NotImplementedException ex = ::sidl::NotImplementedException::_create();
//   ex.setNote("This method has not been implemented");
//   ex.add(__FILE__, __LINE__, "create");
//   throw ex;
  // DO-DELETE-WHEN-IMPLEMENTING exception.end(scijump.EventService.create)

  scijump::EventService es = scijump::EventService::_create();
  es.initialize(framework);
  return es;

  // DO-NOT-DELETE splicer.end(scijump.EventService.create)
}


// user defined non-static methods:
/**
 * Method:  initialize[]
 */
void
scijump::EventService_impl::initialize_impl (
  /* in */::sci::cca::AbstractFramework& framework ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.EventService.initialize)
  this->framework = framework;
  // DO-NOT-DELETE splicer.end(scijump.EventService.initialize)
}

/**
 *  Get a Topic by passing a name that has the form X.Y.Z. The
 * method creates a topic of topicName it if it doesn't exist.
 * 
 * @topicName - A dot delimited, hierarchical name of the topic
 * on which to publish. Wildcard characters are not
 * allowed for a topicName.
 */
::sci::cca::Topic
scijump::EventService_impl::getTopic_impl (
  /* in */const ::std::string& topicName ) 
// throws:
//     ::sci::cca::EventServiceException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.EventService.getTopic)
  if (topicName.empty()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Topic name is empty");
    throw ex;
  }

  //Check if Topic Name has '*'
  for (unsigned int i = 0; i < topicName.size(); i++) {
    if (topicName.at(i) == '*') {
      EventServiceException ex = EventServiceException::_create();
      ex.setNote("Topic Name format not supported:'*' not allowed");
      throw ex;
    }
  }

  Topic topicPtr;
  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) { // new Topic
    topicPtr = Topic::_create();
    topicPtr.initialize(topicName, framework);
    topicMap[topicName] = topicPtr;
  } else { // Topic already present
    topicPtr = iter->second;
  }

  // for all Subscriptions:
  // check for matching topic name and add to newly created Topic's Subscriptions map
  for (SubscriptionMap::iterator subscriptionIter = subscriptionMap.begin();
       subscriptionIter != subscriptionMap.end(); subscriptionIter++) {

    if (isMatch((subscriptionIter->second).getSubscriptionName(), topicName)) {
      if (topicPtr._is_nil()) {
        EventServiceException ex = EventServiceException::_create();
        ex.setNote("Topic Name format not supported:'*' not allowed");
        throw ex;
      }
      topicPtr.addSubscription(topicName, subscriptionIter->second);
    }

  }

  return topicPtr;
  // DO-NOT-DELETE splicer.end(scijump.EventService.getTopic)
}

/**
 *  Returns true if topic already exists, false otherwise 
 */
bool
scijump::EventService_impl::existsTopic_impl (
  /* in */const ::std::string& topicName ) 
{
  // DO-NOT-DELETE splicer.begin(scijump.EventService.existsTopic)
  Topic topicPtr;
  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) { 
    return false;
  } else { 
    return true;
  }
  // DO-NOT-DELETE splicer.end(scijump.EventService.existsTopic)
}

/**
 *  Subscribe to one or more topics.
 * 
 * @subscriptionName - A dot delimited hierarchical name selecting
 * the list of topics to get events from. Wildcard
 * characters (,?)  are allowed for a subscriptionName
 * to denote more than one topic.
 */
::sci::cca::Subscription
scijump::EventService_impl::getSubscription_impl (
  /* in */const ::std::string& subscriptionName ) 
// throws:
//     ::sci::cca::EventServiceException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(scijump.EventService.getSubscription)
  if (subscriptionName.empty()) {
    EventServiceException ex = EventServiceException::_create();
    ex.setNote("Subscription name is empty");
    throw ex;
  }

  Subscription subscriptionPtr;
  SubscriptionMap::iterator iter = subscriptionMap.find(subscriptionName);
  if (iter == subscriptionMap.end()) { // new Subscription
    subscriptionPtr = Subscription::_create(); 
    subscriptionPtr.initialize(subscriptionName, framework);
    subscriptionMap[subscriptionName] = subscriptionPtr;
  } else { // Subscription already present
    subscriptionPtr = iter->second;
  }

  // for all Topics:
  // add newly created Subscription to matching Topic's Subscriptions map
  for (TopicMap::iterator topicIter = topicMap.begin(); topicIter != topicMap.end(); topicIter++) {
    std::string topicName = (topicIter->second).getTopicName();
    if (isMatch(topicName, subscriptionName)) {
      Topic t = topicIter->second;
      if (t._is_nil()) {
	EventServiceException ex = EventServiceException::_create();
	ex.setNote("Topic pointer is null");
	throw ex;
      }
      t.addSubscription(topicName, subscriptionPtr);
    }
  }

  return subscriptionPtr;
  // DO-NOT-DELETE splicer.end(scijump.EventService.getSubscription)
}

/**
 *  Process published events. When the subscriber calls this method,
 * this thread or some other one delivers each event by calling
 * processEvent(...) on each listener belonging to each specific
 * Subscription 
 */
void
scijump::EventService_impl::processEvents_impl () 
// throws:
//     ::sci::cca::EventServiceException
//     ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(scijump.EventService.processEvents)
  if (topicMap.empty()) {
    return;
  }

  for (TopicMap::iterator iter = topicMap.begin(); iter != topicMap.end(); iter++) {
    // Call processEvents() on each Topic
    Topic t = iter->second;
    if (t._is_nil()) {
      EventServiceException ex = EventServiceException::_create();
      ex.setNote("Topic pointer is null");
      throw ex;
    }
    t.processEvents();
  }
  // DO-NOT-DELETE splicer.end(scijump.EventService.processEvents)
}


// DO-NOT-DELETE splicer.begin(scijump.EventService._misc)
bool scijump::EventService_impl::isMatch(const std::string& topicName, 
					 const std::string& subscriptionName)
{
  std::string::size_type s_star = subscriptionName.find('*');

  if(s_star != std::string::npos) {
    std::string word = subscriptionName.substr(0, s_star);
    std::string::size_type t_start = topicName.find(word);
    std::string::size_type s_newstar = 0;
    std::string::size_type t_end = 0;

    if(t_start == 0) {
      while(t_start != std::string::npos) {
        if(subscriptionName.length() == s_star) return true;

        t_end = t_start + word.length();
        s_newstar = subscriptionName.find('*',s_star+1);
        if(s_newstar == std::string::npos) s_newstar = subscriptionName.length();
        word = subscriptionName.substr(s_star+1, s_newstar);
        s_star = s_newstar;

        t_start = topicName.find(word,t_end+1);
      }
    }
  }

  std::string::size_type s_percent = subscriptionName.find('%');
  if(s_percent != std::string::npos) {
    std::string word = subscriptionName.substr(0, s_percent);
    std::string::size_type t_start = topicName.find(word);
    std::string::size_type s_newpercent = 0;
    std::string::size_type t_end = 0;

    if(t_start == 0) {
      while(t_start != std::string::npos) {
        if(subscriptionName.length() == s_percent) return true;

        t_end = t_start + word.length();
        s_newpercent = subscriptionName.find('%',s_percent+1);
        if(s_newpercent == std::string::npos) s_newpercent = subscriptionName.length();
        word = subscriptionName.substr(s_percent+1, s_newpercent);
        s_percent = s_newpercent;

        t_start = topicName.find(word,t_end+1);
        if(t_start != t_end+1) break;
      }
    }

  }

  return false;
}
// DO-NOT-DELETE splicer.end(scijump.EventService._misc)


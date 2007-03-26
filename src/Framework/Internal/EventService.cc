/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/

#include <Framework/Internal/EventService.h>
#include <Framework/Internal/Topic.h>
#include <Framework/Internal/Subscription.h>
#include <Framework/Internal/EventServiceException.h>
#include <Framework/SCIRunFramework.h>
#include <Framework/CCA/CCAException.h>

namespace SCIRun {

EventService::EventService(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:EventService")
{
}

EventService::~EventService()
{
  topicMap.clear();
  subscriptionMap.clear();
}

sci::cca::ComponentID::pointer
EventService::createInstance(const std::string& instanceName,
                             const std::string& className,
                             const sci::cca::TypeMap::pointer& properties)
{
  if (instanceName.size()) {
    if (framework->lookupComponent(instanceName) != 0) {
      throw CCAExceptionPtr(new CCAException("Component instance name " + instanceName + " is not unique"));
    }
    return framework->createComponentInstance(instanceName, className, properties);
  }
  return framework->createComponentInstance(framework->getUniqueName(className), className, properties);
}

InternalFrameworkServiceInstance*
EventService::create(SCIRunFramework* framework)
{
  return new EventService(framework);
}

sci::cca::Topic::pointer EventService::createTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty"));
  }

  //Check if Topic Name has '*'
  for (unsigned int i = 0; i < topicName.size(); i++) {
    if (topicName.at(i) == '*') {
      throw EventServiceExceptionPtr(new EventServiceException("Topic Name format not supported:'*' not allowed"));
    }
  }

  sci::cca::Topic::pointer topicPtr;
  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) { // new Topic
    topicPtr = new Topic(topicName,framework);
    topicMap[topicName] = topicPtr;
  } else { // Topic already present
    topicPtr = iter->second;
  }

  // for all Subscriptions:
  // check for matching topic name and add to newly created Topic's Subscriptions map
  for (SubscriptionMap::iterator subscriptionIter = subscriptionMap.begin();
       subscriptionIter != subscriptionMap.end(); subscriptionIter++) {

    if (isMatch(subscriptionIter->second->getSubscriptionName(), topicName)) {
      Topic* t = dynamic_cast<Topic*>(topicPtr.getPointer());
      if (t == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("Topic pointer is null"));
      }
      t->addSubscription(topicName, subscriptionIter->second);

    }
  }
  return topicPtr;
}

sci::cca::Topic::pointer EventService::getTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty"));
  }

  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found"));
  } else {
    return iter->second;
  }
}

sci::cca::Subscription::pointer EventService::subscribeToEvents(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty"));
  }

  sci::cca::Subscription::pointer subscriptionPtr;
  SubscriptionMap::iterator iter = subscriptionMap.find(topicName);
  if (iter == subscriptionMap.end()) { // new Subscription
    subscriptionPtr = new Subscription(topicName);
    subscriptionMap[topicName] = subscriptionPtr;
  } else { // Subscription already present
    subscriptionPtr = iter->second;
  }

  // for all Topics:
  // add newly created Subscription to matching Topic's Subscriptions map
  for (TopicMap::iterator topicIter = topicMap.begin(); topicIter != topicMap.end(); topicIter++) {
    if (isMatch(topicIter->second->getTopicName(), topicName)) {
      Topic* t = dynamic_cast<Topic*>((topicIter->second).getPointer());
      if (t == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("Topic pointer is null"));
      }
      t->addSubscription(topicName, subscriptionPtr);
    }
  }
  return subscriptionPtr;
}

sci::cca::Subscription::pointer EventService::getSubscription(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty"));
  }

  SubscriptionMap::iterator iter = subscriptionMap.find(topicName);
  if (iter == subscriptionMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found"));
  } else {
    return iter->second;
  }
}

void EventService::releaseTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty"));
  }

  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found"));
  }
  topicMap.erase(iter);
}

void EventService::releaseSubscription(const sci::cca::Subscription::pointer& subscription)
{
  if (subscription.isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty"));
  }

  SubscriptionMap::iterator iter = subscriptionMap.find(subscription->getSubscriptionName());
  if (iter == subscriptionMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found"));
  }

  // for all Topics that match this Subscription: remove this Subscription from the subscriptionsMap
  for (TopicMap::iterator topicIter = topicMap.begin(); topicIter != topicMap.end(); topicIter++) {
    if (isMatch(topicIter->second->getTopicName(), subscription->getSubscriptionName())) {
      Topic* t = dynamic_cast<Topic*>((topicIter->second).getPointer());
      if (t == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("Topic pointer is null"));
      }
      t->removeSubscription(subscription->getSubscriptionName());
    }
  }
  subscriptionMap.erase(iter);
}

void EventService::processEvents()
{
  if (topicMap.empty()) {
    return;
  }

  for (TopicMap::iterator iter = topicMap.begin(); iter != topicMap.end(); iter++) {
    // Call processEvents() on each Topic
    Topic* t = dynamic_cast<Topic*>((iter->second).getPointer());
    if (t == 0) {
      throw EventServiceExceptionPtr(new EventServiceException("Topic pointer is null"));
    }
    t->processEvents();
  }
}


// TODO: check subscriptionName for correct formatting?
// Note: WildcadTopic format needs to decided.
bool EventService::isMatch(const std::string& topicName, const std::string& subscriptionName)
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

}

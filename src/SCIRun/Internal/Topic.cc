/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <SCIRun/Internal/Topic.h>
#include <SCIRun/Internal/WildcardTopic.h>
#include <SCIRun/Internal/EventServiceException.h>

namespace SCIRun {

Topic::Topic(const std::string& name,SCIRunFramework *fwk) : topicName(name),framework(fwk) {}

Topic::~Topic()
{
  eventList.clear();
  eventListenerMap.clear();
}
void Topic::sendEvent(const sci::cca::Event::pointer& event)
{
  std::string fwkURL = framework->getURL().getString();
//   std::string name = framework->getUniqueName(fwkURL);
//   std::cout << "Unique name : " << name << std::endl;
//   std::cout << "Fwk URL : " << fwkURL << std::endl;
  sci::cca::TypeMap::pointer eventHeader = event->getHeader();
  sci::cca::TypeMap::pointer eventBody = event->getBody();
  if (event->getHeader().isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("eventHeader pointer is null", sci::cca::Unexpected));
  }

  if (event->getBody().isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("eventBody pointer is null", sci::cca::Unexpected));
  }
  SSIDL::array1<std::string> allKeys = eventHeader->getAllKeys(sci::cca::None);
  bool isURLPresent = false;
  for (unsigned int i = 0; i < allKeys.size(); i++){
    if(allKeys[i]  == "FrameworkURL"){
      //std::cout << "Framework data present in Event Header\n";
      isURLPresent = true;
      break;
    }
  }
  if(!isURLPresent){
    //std::cout << "Framework data not present in Event Header, so adding one\n";
    eventHeader->putString(std::string("FrameworkURL"),fwkURL);
  }
  eventList.push_back(event);
}

void Topic::registerEventListener(const std::string &listenerKey, const sci::cca::EventListener::pointer &theListener)
{
  if (listenerKey.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener Key is empty", sci::cca::Unexpected));
  }

  if (theListener.isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener Pointer is null", sci::cca::Unexpected));
  }

  EventListenerMap::iterator iter = eventListenerMap.find(listenerKey);
  if (iter != eventListenerMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener Key already present", sci::cca::Unexpected));
  }
  eventListenerMap[listenerKey] = theListener;
}

void Topic::unregisterEventListener(const std::string &listenerKey)
{
  if (listenerKey.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener Key is empty", sci::cca::Unexpected));
  }

  EventListenerMap::iterator iter = eventListenerMap.find(listenerKey);
  if (iter == eventListenerMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener Key not found", sci::cca::Unexpected));
  }
  eventListenerMap.erase(iter);
}

void Topic::processEvents()
{
  for (EventListenerMap::iterator eventListenerIter = eventListenerMap.begin();
       eventListenerIter != eventListenerMap.end();
       eventListenerIter++) {
    // Call processEvent() for each Listener
    for (unsigned int i = 0; i < eventList.size(); i++) {
      // Call processEvent() for each event
      eventListenerIter->second->processEvent(topicName, eventList[i]);
    }
  }

  // Call processEvents(..) for each WildcardTopic
  for (WildcardTopicMap::iterator wildcardTopicIter = wildcardTopicMap.begin();
       wildcardTopicIter != wildcardTopicMap.end(); wildcardTopicIter++) {

    WildcardTopic *wildcardTopicPtr = dynamic_cast<WildcardTopic*>((wildcardTopicIter->second).getPointer());
    if (wildcardTopicPtr == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("WildcardTopic pointer is null"));
    }
    wildcardTopicPtr->processEvents(eventList);
  }
  eventList.clear();
}

void Topic::addWildcardTopic(const std::string& topicName, const sci::cca::WildcardTopic::pointer &theWildcardTopic)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  if (theWildcardTopic.isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("WildcardTopic pointer is null", sci::cca::Unexpected));
  }

  WildcardTopicMap::iterator iter =  wildcardTopicMap.find(topicName);
  if (iter != wildcardTopicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("WildcardTopic already present", sci::cca::Unexpected));
  }
  wildcardTopicMap[topicName] = theWildcardTopic;
}

void Topic::removeWildcardTopic(const std::string& topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  WildcardTopicMap::iterator iter = wildcardTopicMap.find(topicName);
  if (iter == wildcardTopicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("WildcardTopic not found", sci::cca::Unexpected));
  }
  wildcardTopicMap.erase(iter);
}

}

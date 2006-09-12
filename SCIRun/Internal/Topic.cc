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

#include <SCIRun/Internal/Topic.h>
#include <SCIRun/Internal/WildcardTopic.h>
#include <SCIRun/Internal/EventServiceException.h>
#include <Core/Thread/AtomicCounter.h>
namespace SCIRun {

Topic::Topic(const std::string& name,const sci::cca::ports::EventService::pointer& esPtr) : topicName(name),eventServicePtr(esPtr) {}

Topic::~Topic()
{
  eventList.clear();
  eventListenerMap.clear();
}
void Topic::sendEvent(const sci::cca::Event::pointer& event)
{
  sci::cca::TypeMap::pointer eventHeader = event->getHeader();
  sci::cca::TypeMap::pointer eventBody = event->getBody();
  if (event->getHeader().isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("eventHeader pointer is null", sci::cca::Unexpected));
  }

  if (event->getBody().isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("eventBody pointer is null", sci::cca::Unexpected));
  }
  EventService *ptr = dynamic_cast<EventService*>(eventServicePtr.getPointer());
  std::string fwkURL = ptr->getFrameworkURL();
  int count = ptr->incrementCount();
  SSIDL::array1<std::string> allKeys = eventHeader->getAllKeys(sci::cca::None);
  bool isURLPresent = false;
  for (unsigned int i = 0; i < allKeys.size(); i++){
    if(allKeys[i]  == "FrameworkURL"){
      isURLPresent = true;
      break;
    }
  }
  if(!isURLPresent){
    eventHeader->putString(std::string("FrameworkURL"),fwkURL);
  }
  eventHeader->putInt(std::string("UniqueID"),count);
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
      //Validate Event Header and body
      sci::cca::TypeMap::pointer eventHeader = eventList[i]->getHeader();
      sci::cca::TypeMap::pointer eventBody = eventList[i]->getBody();
      if(eventHeader.isNull()){
        throw EventServiceExceptionPtr(new EventServiceException("Event Header is Null", sci::cca::Unexpected));
      }
      if(eventBody.isNull()){
        throw EventServiceExceptionPtr(new EventServiceException("Event Body is Null", sci::cca::Unexpected));
      }
      // Check if Framework Data is present in Header
      SSIDL::array1<std::string> allKeys = eventHeader->getAllKeys(sci::cca::None);
      bool isURLPresent = false;
      bool isUniqueIDPresent = false;
      for (unsigned int j = 0; j < allKeys.size(); j++){
        if(allKeys[j]  == "FrameworkURL"){
          isURLPresent = true;
        }
        if(allKeys[j]  == "UniqueID"){
          isUniqueIDPresent = true;
        }
      }
      if((!isURLPresent)||(!isUniqueIDPresent)){
        throw EventServiceExceptionPtr(new EventServiceException("Header Info missing", sci::cca::Unexpected));
      }
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

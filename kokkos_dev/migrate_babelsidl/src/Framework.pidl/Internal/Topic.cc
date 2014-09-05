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

#include <Framework/Internal/Topic.h>
#include <Framework/Internal/Subscription.h>
#include <Framework/Internal/EventServiceException.h>

namespace SCIRun {

Topic::Topic(const std::string& name,SCIRunFramework *fwk) : topicName(name),framework(fwk) {}

Topic::~Topic()
{
  eventList.clear();
  eventListenerMap.clear();
}

void Topic::sendEvent(const std::string& topicName, const sci::cca::TypeMap::pointer& theBody)
{
  std::string fwkURL = framework->getURL().getString();
//   std::string name = framework->getUniqueName(fwkURL);
//   std::cout << "Unique name : " << name << std::endl;
//   std::cout << "Fwk URL : " << fwkURL << std::endl;


  sci::cca::TypeMap::pointer eventHeader = framework->createTypeMap();
  sci::cca::Event::pointer event(new Event(eventHeader,theBody));

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

  // Call processEvents(..) for each Subscription
  for (SubscriptionMap::iterator subscriptionIter = subscriptionMap.begin();
       subscriptionIter != subscriptionMap.end(); subscriptionIter++) {

    Subscription *subscriptionPtr = dynamic_cast<Subscription*>((subscriptionIter->second).getPointer());
    if (subscriptionPtr == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("Subscription pointer is null"));
    }
    subscriptionPtr->processEvents(eventList);
  }
  eventList.clear();
}

void Topic::addSubscription(const std::string& topicName, const sci::cca::Subscription::pointer &theSubscription)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  if (theSubscription.isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("Subscription pointer is null", sci::cca::Unexpected));
  }

  SubscriptionMap::iterator iter =  subscriptionMap.find(topicName);
  if (iter != subscriptionMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Subscription already present", sci::cca::Unexpected));  }
  subscriptionMap[topicName] = theSubscription;
}

void Topic::removeSubscription(const std::string& topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  SubscriptionMap::iterator iter = subscriptionMap.find(topicName);
  if (iter == subscriptionMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Subscription not found", sci::cca::Unexpected));
  }
  subscriptionMap.erase(iter);
}

}

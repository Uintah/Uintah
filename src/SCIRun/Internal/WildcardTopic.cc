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

#include <SCIRun/Internal/WildcardTopic.h>
#include <SCIRun/Internal/EventServiceException.h>

namespace SCIRun {

WildcardTopic::WildcardTopic(const std::string& name) : topicName(name)
{
}

WildcardTopic::~WildcardTopic()
{
  eventListenerMap.clear();
}

void WildcardTopic::registerEventListener(const std::string &listenerKey, const sci::cca::EventListener::pointer& theListener)
{
  if (listenerKey.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener key is empty", sci::cca::Unexpected));
  }

  if (theListener.isNull()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener pointer is null", sci::cca::Unexpected));
  }
  EventListenerMap::iterator iter =  eventListenerMap.find(listenerKey);
  if (iter != eventListenerMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener key already present", sci::cca::Unexpected));
  }
  eventListenerMap[listenerKey] = theListener;
}

void WildcardTopic::unregisterEventListener(const std::string& listenerKey)
{
  if (listenerKey.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener key is empty", sci::cca::Unexpected));
  }

  EventListenerMap::iterator iter =  eventListenerMap.find(listenerKey);
  if (iter == eventListenerMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Listener key not found", sci::cca::Unexpected));
  }
  eventListenerMap.erase(iter);
}

void WildcardTopic::processEvents(const EventPtrList& eventList)
{
  if (eventListenerMap.empty()) {
    return;
  }

  if (eventList.empty()) {
    return;
  }

  for (EventListenerMap::iterator eventListenerIter = eventListenerMap.begin();
       eventListenerIter != eventListenerMap.end(); eventListenerIter++) {

    // Call processEvent() on each Listener
    for (unsigned int i = 0; i < eventList.size(); i++) {
      // Call processEvent() on each event
      eventListenerIter->second->processEvent(topicName, eventList[i]);
    }
  }
}

}

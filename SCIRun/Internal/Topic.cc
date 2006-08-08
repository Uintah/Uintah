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

#include <SCIRun/Internal/EventService.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <SCIRun/PortInstanceIterator.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>
#include <SCIRun/PortInstance.h>
#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/CCA/CCAComponentInstance.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/CCA/ConnectionID.h>
#include <SCIRun/Internal/ConnectionEvent.h>
#include <SCIRun/Internal/ConnectionEventService.h>
#include <SCIRun/Internal/EventServiceException.h>
#include <iostream>
#include <string>
#include <vector>
namespace SCIRun {



Topic::Topic(std::string name) : topicName(name){}

Topic::~Topic()
{
  eventBodyList.clear();
  eventListenerMap.clear();
}

void Topic::sendEvent(const sci::cca::TypeMap::pointer &eventBody)
{
  if(eventBody.isNull())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("EventBody pointer is null", sci::cca::Unexpected)); 
    }
  eventBodyList.push_back(eventBody);
  //std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator wildcardTopicIter;
  // for(wildcardTopicIter = wildcardTopicMap.begin(); wildcardTopicIter != wildcardTopicMap.end(); wildcardTopicIter++){
//     wildcardTopicIter->second->addEvent(eventBody);
//     wildcardTopicIter->second->eventBodyList.push_back(eventBody);
//   }
}
void Topic::registerEventListener(const std::string &listenerKey, const sci::cca::IEventListener::pointer &theListener)
{
  if(listenerKey.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Listener Key is empty", sci::cca::Unexpected)); 
    }
  if(theListener.isNull())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Listener Pointer is null", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::IEventListener::pointer>::iterator iter =  eventListenerMap.find(listenerKey);
  if(iter != eventListenerMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Listener Key already present", sci::cca::Unexpected)); 
    }
  eventListenerMap[listenerKey] = theListener;
}
void Topic::unregisterEventListener(const std::string &listenerKey)
{
  std::map<std::string, sci::cca::IEventListener::pointer>::iterator iter =  eventListenerMap.find(listenerKey);
  if(listenerKey.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Listener Key is empty", sci::cca::Unexpected)); 
    }
  if(iter == eventListenerMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Listener Key not found", sci::cca::Unexpected)); 
    }
  eventListenerMap.erase(iter);
}

void Topic::processEvents()
{
//   if(eventListenerMap.empty()||eventBodyList.empty())
//     {
//       return;
//     }
  std::map<std::string, sci::cca::IEventListener::pointer>::iterator eventListenerIter;
  for(eventListenerIter = eventListenerMap.begin(); eventListenerIter != eventListenerMap.end(); eventListenerIter++)
    {
      //Call processEvent() for each Listener
      for(unsigned int i = 0; i<eventBodyList.size(); i++)
        {
          //Call processEvent() for each event
          eventListenerIter->second->processEvent(topicName,eventBodyList[i]);
        }
     }
   //loop thro' all wildcard topics and call process events on each of those
  std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator wildcardTopicIter;
  for(wildcardTopicIter = wildcardTopicMap.begin(); wildcardTopicIter != wildcardTopicMap.end(); wildcardTopicIter++){
    //wildcardTopicIter->second->processEvents(eventListenerMap);
    WildcardTopic *wildcardTopicPtr = dynamic_cast<WildcardTopic*>(wildcardTopicIter->second.getPointer());
    wildcardTopicPtr->processEvents(eventBodyList);
  }
  eventBodyList.clear();
}
std::string Topic::getTopicName()
{
  return topicName;
}

void Topic::addWildcardTopic(const std::string& topicName,const sci::cca::WildcardTopic::pointer &theWildcardTopic)
{
    if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }

    if(theWildcardTopic.isNull())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("WildcardTopic pointer is null", sci::cca::Unexpected)); 
    }
    std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator iter =  wildcardTopicMap.find(topicName);
    if(iter != wildcardTopicMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("WildcardTopic already present", sci::cca::Unexpected)); 
    }
    wildcardTopicMap[topicName] = theWildcardTopic;
}
void Topic::removeWildcardTopic(const std::string& topicName)
{
  if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator iter = wildcardTopicMap.find(topicName);
  if(iter == wildcardTopicMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("WildcardTopic not found", sci::cca::Unexpected)); 
    }
  wildcardTopicMap.erase(iter);
}
}

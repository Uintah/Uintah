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


EventService::EventService(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:EventService")
{
}

EventService::~EventService()
{
//   eventMap.clear();
//   eventListenerMap.clear();
//   topicList.clear();
  topicMap.clear();
}

sci::cca::ComponentID::pointer
EventService::createInstance(const std::string& instanceName,
                               const std::string& className,
                               const sci::cca::TypeMap::pointer& properties)
{
  if (instanceName.size()) {
    if (framework->lookupComponent(instanceName) != 0) {
      throw sci::cca::CCAException::pointer(new CCAException("Component instance name " + instanceName + " is not unique"));
    }
    return framework->createComponentInstance(instanceName, className, properties);
  }
  return framework->createComponentInstance(framework->getUniqueName(className), className, properties);
}

InternalFrameworkServiceInstance*
EventService::create(SCIRunFramework* framework)
{
  EventService* n = new EventService(framework);
  return n;
}
sci::cca::Port::pointer
EventService::getService(const std::string&)
{
  return sci::cca::Port::pointer(this);
}

sci::cca::Topic::pointer EventService::getTopic(const std::string &topicName)
{
  if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::Topic::pointer>::iterator iter = topicMap.find(topicName);
  if(iter == topicMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic not found", sci::cca::Unexpected)); 
    }
  else
    {
       sci::cca::Topic::pointer topicPtr(iter->second);
       return topicPtr;
    }

}
sci::cca::WildcardTopic::pointer EventService::getWildcardTopic(const std::string &topicName)
{
  if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator iter = wildcardTopicMap.find(topicName);
  if(iter == wildcardTopicMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic not found", sci::cca::Unexpected)); 
    }
  else
    {
       sci::cca::WildcardTopic::pointer wildcardTopicPtr(iter->second);
       return wildcardTopicPtr;
    }

}
sci::cca::Topic::pointer EventService::createTopic(const std::string &topicName)
{
  sci::cca::Topic::pointer topicPtr;
  if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::Topic::pointer>::iterator iter = topicMap.find(topicName);
  //Topic not present previously
  if(iter == topicMap.end())
    {
      topicPtr = new Topic(topicName);
      topicMap[topicName] = topicPtr;
    }
  
  //Topic already present
  else{
     topicPtr = iter->second;
  }  
  //For all wildcard topics check is the match topic name and add to wildcardtopics map.
  std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator wildcardTopicIter;
  for(wildcardTopicIter = wildcardTopicMap.begin(); wildcardTopicIter != wildcardTopicMap.end(); wildcardTopicIter++)
    if(isMatch(wildcardTopicIter->second->getTopicName(),topicName))
      topicPtr->addWildcardTopic(topicName,wildcardTopicIter->second);
  return topicPtr;
}
sci::cca::WildcardTopic::pointer EventService::createWildcardTopic(const std::string &topicName)
{
  sci::cca::WildcardTopic::pointer wildcardTopicPtr;
  if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator iter = wildcardTopicMap.find(topicName);
  //Topic not present previously
  if(iter == wildcardTopicMap.end())
    {
      wildcardTopicPtr = new WildcardTopic(topicName);
      wildcardTopicMap[topicName] = wildcardTopicPtr;
    }
  else{
  //Topic already present
      wildcardTopicPtr = iter->second;
  }
  //For all topics add to respective topic's wildcardtopics map
  std::map<std::string, sci::cca::Topic::pointer>::iterator topicIter;
  for(topicIter = topicMap.begin(); topicIter != topicMap.end(); topicIter++)
    if(isMatch(topicIter->second->getTopicName(),topicName)){
      topicIter->second->addWildcardTopic(topicName,wildcardTopicPtr);
    }
return wildcardTopicPtr;

}
void EventService::releaseTopic(const std::string &topicName)
{
  if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::Topic::pointer>::iterator iter = topicMap.find(topicName);
  if(iter == topicMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic not found", sci::cca::Unexpected)); 
    }
  topicMap.erase(iter);
}
void EventService::releaseWildcardTopic(const std::string &topicName)
{
  if(topicName.empty())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic name empty", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::WildcardTopic::pointer>::iterator iter = wildcardTopicMap.find(topicName);
  if(iter == wildcardTopicMap.end())
    {
      throw sci::cca::EventServiceException::pointer (new EventServiceException("Topic not found", sci::cca::Unexpected)); 
    }
  std::map<std::string, sci::cca::Topic::pointer>::iterator topicIter;
  //for all topics that match this wildcard topic remove this wildcartopic from the wildcatdtopics map
  for(topicIter = topicMap.begin(); topicIter != topicMap.end(); topicIter++)
    if(isMatch(topicIter->second->getTopicName(),topicName))
      topicIter->second->removeWildcardTopic(topicName);
  wildcardTopicMap.erase(iter);
}
void EventService::processEvents()
{
  std::map<std::string, sci::cca::Topic::pointer>::iterator iter;
  if(topicMap.empty())
    {
      return;
    }
  for(iter = topicMap.begin(); iter != topicMap.end(); iter++)
    {
      //Call processEvents() for each topic
      iter->second->processEvents();
    }
  
}
bool EventService::isMatch(std::string topicName, std::string wildcardTopicName)
{
  std::string::size_type wildcard  = wildcardTopicName.find('*');
  if (wildcard == std::string::npos)
    return false;
  std::string wc = wildcardTopicName.substr(0,wildcard);
  std::string::size_type loc = topicName.find(wc);
  if (loc == 0)
    return true;
  else
    return false;
}
}


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
#include <SCIRun/Internal/Topic.h>
#include <SCIRun/Internal/WildcardTopic.h>
#include <SCIRun/Internal/EventServiceException.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/CCAException.h>
#include <Core/Thread/AtomicCounter.h>

namespace SCIRun {

EventService::EventService(SCIRunFramework* framework)
  : InternalFrameworkServiceInstance(framework, "internal:EventService"),atomicCount("Event")
{

}

EventService::~EventService()
{
  topicMap.clear();
  wildcardTopicMap.clear();
}

sci::cca::ComponentID::pointer
EventService::createInstance(const std::string& instanceName,
                             const std::string& className,
                             const sci::cca::TypeMap::pointer& properties)
{
  if (instanceName.size()) {
    if (framework->lookupComponent(instanceName) != 0) {
      //throw CCAExceptionPtr(new CCAException("Component instance name " + instanceName + " is not unique"));
      throw sci::cca::CCAException::pointer (new CCAException("Component instance name " + instanceName + " is not unique"));
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
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  //Check if Topic Name has '*'
  for(unsigned int i=0;i<topicName.size();i++)
    if(topicName.at(i) == '*')
      throw EventServiceExceptionPtr(new EventServiceException("Topic Name format not supported:'*' not allowed", sci::cca::Unexpected));

  sci::cca::Topic::pointer topicPtr;
  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) { // new Topic
    sci::cca::ports::EventService::pointer eventServicePtr(this);
    topicPtr = new Topic(topicName,eventServicePtr);
    topicMap[topicName] = topicPtr;
  } else { // Topic already present
    topicPtr = iter->second;
  }

  // for all WildcardTopics:
  // check for matching topic name and add to newly created Topic's WildcardTopics map
  for (WildcardTopicMap::iterator wildcardTopicIter = wildcardTopicMap.begin();
       wildcardTopicIter != wildcardTopicMap.end(); wildcardTopicIter++) {

    if (isMatch(wildcardTopicIter->second->getTopicName(), topicName)) {
      Topic* t = dynamic_cast<Topic*>(topicPtr.getPointer());
      if (t == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("Topic pointer is null"));
      }
      t->addWildcardTopic(topicName, wildcardTopicIter->second);

    }
  }
  return topicPtr;
}

sci::cca::Topic::pointer EventService::getTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found", sci::cca::Unexpected));
  } else {
    return iter->second;
  }
}

sci::cca::WildcardTopic::pointer EventService::createWildcardTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }


  //Check if wildcard Topic Name doesn't have any wildcard characters
  if((topicName.find('*')==std::string::npos)&&(topicName.find('?') == std::string::npos)){
    std::cout << "WARNING!!! - WildcardTopicName doesn't contain any Wildcard characters\n";
  }


  //Check if topicName has more than one '*' s
  int countStars = 0;
  for(unsigned int i=0;i<topicName.size();i++)
    if(topicName.at(i) == '*')
      countStars++;
  if(countStars > 1){
    throw EventServiceExceptionPtr(new EventServiceException("WildcardTopic Name format not supported: More than one '*' not allowed", sci::cca::Unexpected));
  }

  sci::cca::WildcardTopic::pointer wildcardTopicPtr;
  WildcardTopicMap::iterator iter = wildcardTopicMap.find(topicName);
  if (iter == wildcardTopicMap.end()) { // new WildcardTopic
    wildcardTopicPtr = new WildcardTopic(topicName);
    wildcardTopicMap[topicName] = wildcardTopicPtr;
  } else { // WildcardTopic already present
    wildcardTopicPtr = iter->second;
  }

  // for all Topics:
  // add newly created WildcardTopic to matching Topic's WildcardTopics map
  for (TopicMap::iterator topicIter = topicMap.begin(); topicIter != topicMap.end(); topicIter++) {
    if (isMatch(topicIter->second->getTopicName(), topicName)) {
      Topic* t = dynamic_cast<Topic*>((topicIter->second).getPointer());
      if (t == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("Topic pointer is null"));
      }
      t->addWildcardTopic(topicName, wildcardTopicPtr);
    }
  }
  return wildcardTopicPtr;
}

sci::cca::WildcardTopic::pointer EventService::getWildcardTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  WildcardTopicMap::iterator iter = wildcardTopicMap.find(topicName);
  if (iter == wildcardTopicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found", sci::cca::Unexpected));
  } else {
    return iter->second;
  }
}

void EventService::releaseTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  TopicMap::iterator iter = topicMap.find(topicName);
  if (iter == topicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found", sci::cca::Unexpected));
  }
  topicMap.erase(iter);
}

void EventService::releaseWildcardTopic(const std::string &topicName)
{
  if (topicName.empty()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic name empty", sci::cca::Unexpected));
  }

  WildcardTopicMap::iterator iter = wildcardTopicMap.find(topicName);
  if (iter == wildcardTopicMap.end()) {
    throw EventServiceExceptionPtr(new EventServiceException("Topic not found", sci::cca::Unexpected));
  }

  // for all Topics that match this WildcardTopic: remove this WildcardTopic from the wildcardTopicsMap
  for (TopicMap::iterator topicIter = topicMap.begin(); topicIter != topicMap.end(); topicIter++) {
    if (isMatch(topicIter->second->getTopicName(), topicName)) {
      Topic* t = dynamic_cast<Topic*>((topicIter->second).getPointer());
      if (t == 0) {
        throw EventServiceExceptionPtr(new EventServiceException("Topic pointer is null"));
      }
      t->removeWildcardTopic(topicName);
    }
  }
  wildcardTopicMap.erase(iter);
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


// TODO: check wildcardTopicName for correct formatting?
// Note: WildcadTopic format needs to decided.

bool EventService::isPresent(const std::string& givenName, const std::string& topicName)
{
  
  for(unsigned int i=0;i< givenName.size();i++){
    //Check if current charecter is '?' 
    if(givenName.at(i) == '?'){
      continue;
    }
    //Check if current character is equal to corresponding character in topicName
    if (givenName.at(i) != topicName.at(i)){
      return false;
    }
  }
  return true;
}
bool EventService::isMatch(const std::string& topicName, const std::string& wildcardTopicName)
{
  bool match = false;

  if(isPresent(wildcardTopicName,topicName))
    match = true;
  std::string::size_type wildcardTokenPos = wildcardTopicName.find('*');
  if (wildcardTokenPos != std::string::npos){
    //wildcardBeg is a substring of wildcardTopicName upto the location of '*'
    std::string wildcardBeg(wildcardTopicName.substr(0, wildcardTokenPos));
    //wildcardEnd is a substring of wildcardTopicName from the location of '*' till the end
    std::string wildcardEnd(wildcardTopicName.substr(wildcardTokenPos+1));
    //topicBeg is a  substring of topicName upto 'wildcardBeg.size()' characters from the beginning
    std::string topicBeg(topicName.substr(0, wildcardTokenPos));
    //topicEnd is a substring of topicName with 'wildcardTopicEnd.size()' characters counting from the end.
    std::string topicEnd(topicName.substr(topicName.size()-wildcardEnd.size()));
    if((isPresent(wildcardBeg,topicBeg))&&(isPresent(wildcardEnd,topicEnd)))
      match = true;
  }
  return match;
}

std::string EventService::getFrameworkURL()
{
    return framework->getURL().getString();
}

int EventService::incrementCount()
{
  atomicCount++;
  return atomicCount;
}
}

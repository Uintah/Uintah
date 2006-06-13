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
  eventMap.clear();
  eventListenerMap.clear();
  topicList.clear();
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
/*** Just one string ***/

// void EventService::sendEvent(const std::string &eventName, const std::string &eventBody)
// {
//   if(eventName.empty())
//     {
//       std::cout << "Event Name is Empty\n";
//       return;
//     }
//    if(eventBody.empty())
//     {
//       std::cout << "Event Body is Empty\n";
//       return;
//     }
//     eventMap[eventName]=eventBody;
  
// }

// void EventService::receiveEvent(const std::string &eventName,std::string &eventBody)
// {
//   if(eventName.empty())
//     {
//       std::cout << "Event Name is Empty\n";
//       return;
//     }
//   std::map<std::string, std::string>::iterator iter = eventMap.find(eventName);
//   if (iter != eventMap.end()) {
//    eventBody = iter->second;
//    } else {
//     eventBody = std::string();
//     std::cout << "Event does not exist\n";
//   }
// }

/*** List of Strings and IEventListener*********/

int EventService::createTopic(const std::string &topicName)
{
  //SSIDL::array1<std::string>::iterator iter=listofTopic.Find(topicName);
  for(unsigned int i=0;i<topicList.size();i++)
    {
      if(topicList[i] == topicName)
	{
	  std::cout << "Topic already present\n";
	  return 1;
	}
    }
 topicList.push_back(topicName);
 std::cout << "New Topic created\n";
 return 0;
}
int EventService::destroyTopic(const std::string &topicName)
{
  if(topicName.empty())
    {
      std::cout << "Topic Name empty\n";
      return 1;
    }
  SSIDL::array1<std::string>::iterator iter=std::find(topicList.begin(),topicList.end(),topicName);
  if(iter == topicList.end())
    {
      std::cout << "Topic not present\n";
      return 1;
    }
  topicList.erase(iter);
  std::cout << "Topic destroyed\n";
  return 0;
}
// void processEvents()
// {
//   for(each topic in list of topics)
//     look in the event map for that topic
//       if(eventmap[topic].isEmpty)
// 	continue;
//       else
// 	for each event in the eventmap[topic]
// 	{
// 	  for each listener in eventListenermap[topic]
// 	    call process event for that listener
// 		     }
//         eventmap[topic].clear

 
// }
int EventService::sendEvent(const std::string &topicName, const std::string &eventBody)
 {
   if(topicName.empty())
     {
       std::cout << "Topic Name is Empty\n";
       return 1;
     }
   if(eventBody.empty())
     {
       std::cout << "Event Body is Empty\n";
       return 1;
     }
   SSIDL::array1<std::string>::iterator topicsListIter=std::find(topicList.begin(),topicList.end(),topicName);
    if(topicsListIter == topicList.end())
      {
	std::cout << "Topic doesn't exist\n";
	return 1;
      }
    std::map<std::string, SSIDL::array1<std::string> >::iterator iter = eventMap.find(topicName);
    if(iter != eventMap.end()) {
      iter->second.push_back(eventBody);
    } else {
      SSIDL::array1 <std::string> templist;
      templist.push_back(eventBody);   
      eventMap[topicName]=templist;
    }
    std::cout << "New Event sent\n";
    return 0;
 }

/***** List of strings ********/

// void EventService::receiveEvent(const std::string &topicName,SSIDL::array1<std::string> &eventBodyList)
// {
//   std::map<std::string, SSIDL::array1 <std::string> >::iterator iter = eventMap.find(topicName);      
//   if (iter != eventMap.end()) {
//    eventBodyList = iter->second;
//    } else {
//     std::cout << "Topic not present\n";
//    }
// }

/****** IEventListener Case **********/

// void EventService::registerEventListener(const std::string &topicName,sci::cca::IEventListener::pointer &theListener)
// {
//   SSIDL::array1 <std::string> templist;
//   std::map<std::string, SSIDL::array1 <std::string> >::iterator iter = eventMap.find(topicName);
//   if (iter != eventMap.end()) {
//     templist=iter->second;
//   } else {
//     std::cout << "Topic not present\n";
//   }
//   SSIDL::array1<std::string>::iterator listIter;
//   for(listIter=templist.begin();listIter!=templist.end();listIter++)
//     {
//       theListener->processEvent(topicName,std::string(*listIter));
//     }
// }

int EventService::registerEventListener(const std::string &topicName,sci::cca::IEventListener::pointer &theListener)
{
   if(topicName.empty())
     {
       std::cout << "Topic Name is Empty\n";
       return 1;
     }
   if(theListener.isNull())
     {
       std::cout << "Event Listener pointer is Null!!\n";
       return 1;
     }
   SSIDL::array1<std::string>::iterator topicsListIter=std::find(topicList.begin(),topicList.end(),topicName);
    if(topicsListIter == topicList.end())
      {
	std::cout << "Topic doesn't exist\n";
	return 1;
      }
   std::map<std::string, SSIDL::array1<sci::cca::IEventListener::pointer> >::iterator iter = eventListenerMap.find(topicName);
    if(iter != eventListenerMap.end()) {
      iter->second.push_back(theListener);
    } else {
      SSIDL::array1 <sci::cca::IEventListener::pointer> templist;
      templist.push_back(theListener);
      eventListenerMap[topicName]=templist;
    }
    std::cout << "Event Listener registered\n";
    return 0;
}
void EventService::processEvents()
{
  if(topicList.empty())
    {
      std::cout << "Topic List is empty\n";
      return;
    }
  for(unsigned int i=0;i<topicList.size();i++)
    {
      std::string topic=topicList[i];
      
      //Get the list of events for this topic
      std::map<std::string, SSIDL::array1<std::string> >::iterator eventMapIter = eventMap.find(topic);
      if(eventMapIter != eventMap.end())
	{
	  SSIDL::array1<std::string> eventsList = eventMapIter->second;
	  //Get the list of listeners for this topic
	  std::map<std::string, SSIDL::array1<sci::cca::IEventListener::pointer> >::iterator eventListenerMapIter = eventListenerMap.find(topic);
	  if(eventListenerMapIter != eventListenerMap.end())
	    {
	      SSIDL::array1<sci::cca::IEventListener::pointer> listenersList = eventListenerMapIter->second;
	      std::cout << "Number of Listeners: " << listenersList.size() << std::endl;
	      for(unsigned int i=0;i<listenersList.size();i++)
		for(unsigned int j=0;j<eventsList.size();j++)
		  {
		     listenersList[i]->processEvent(topic,eventsList[j]);
		  }
	    }
	  else
	    {
	      std::cout << "No listeners for this Topic\n";
	      continue;
	    }
	  eventMap.erase(eventMapIter);
	}
      else
	{
	  std::cout << "No Events pending for the Topic: " << topic << std::endl;
	  continue;
	}
      
    }
}
}

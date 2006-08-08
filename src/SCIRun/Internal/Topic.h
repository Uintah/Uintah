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

#ifndef SCIRun_Topic_h
#define SCIRun_Topic_h

#include <sci_metacomponents.h>

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>
#include<Core/CCA/SSIDL/array.h>
#include <list>
#include <vector>
namespace SCIRun {

/* \class Topic */

/* A Topic can be considered as a channel for sending events and listening to events. */
/* The listeners to \em Topic s should implements the  IEventListener interface */

class Topic: public sci::cca::Topic
{
 public:
   Topic(std::string);
   virtual ~Topic();
   
/*    Sends an Event with the specified /em eventBody. The parameter /em eventBody */
/*    is a pointer to a CCA TypeMap of the /em message to be sent */
   virtual void sendEvent(const sci::cca::TypeMap::pointer &eventBody);

/*    Adds a /em listener to the list of listeners for this Topic. The parameter  */
/*    /em listenerKey is used as an index to the list(which is actually a c++ map) */
/*    and the parameter theListener is a pointer to the /em Listener class */
   virtual void registerEventListener(const std::string &listenerKey, const sci::cca::IEventListener::pointer &theListener);
   
/*    Removes a listener from the list of listeners for this Topic. The parameter listenerKey */
/*    is used as an index. */
   virtual void unregisterEventListener(const std::string &listenerKey);
   
/*    Returns the /em topicName for this Topic */
   virtual std::string getTopicName();

/*    The following methods  should be called only by the EventService */
   
/*    Iterates through all the /em listeners for this Topic and calls processEvents()  */
/*    of each listener for each /em Event */
   void processEvents();

/*    Adds a WildcardTopic to the list of Wildcard Topics that correspond to this Topic */
/*    The Parameter /em theWildcardTopic is a pointer to the WildcardTopic that is to be added */
   virtual void addWildcardTopic(const std::string& topicName,const sci::cca::WildcardTopic::pointer &theWildcardTopic);
   
/*    Removes a WildcardTopic from the list of WildcardTopics */
   virtual void removeWildcardTopic(const std::string& topicName);
private:
  std::string topicName;
  SSIDL::array1<sci::cca::TypeMap::pointer> eventBodyList;
  std::map<std::string, sci::cca::IEventListener::pointer> eventListenerMap;
  std::map<std::string, sci::cca::WildcardTopic::pointer> wildcardTopicMap;
};
}
#endif

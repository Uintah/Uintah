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

#ifndef SCIRun_WildcardTopic_h
#define SCIRun_WildcardTopic_h

#include <sci_metacomponents.h>

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>
#include<Core/CCA/SSIDL/array.h>
#include <SCIRun/Internal/EventService.h>
#include <SCIRun/Internal/Topic.h>
#include <list>
#include <vector>
namespace SCIRun {

/* \class WildcardTopic */

/* A WildcardTopic is a collection of Topics. One WildcardTopic may correspond to multiple */
/* \em regular Topics. All listeners to WildcardTopics should also implement  the */
/* IEventListener interface */
class WildcardTopic: public sci::cca::WildcardTopic
{
 public:
   
   virtual ~WildcardTopic();

/* Adds a /em listener to the list of listeners for this WildcardTopic. The parameter */
/* /em listenerKey is used as an index to the list(which is actually a c++ map) */
/* and the parameter theListener is a pointer to the /em Listener class */
   virtual void registerEventListener(const std::string &listenerKey, const sci::cca::IEventListener::pointer &theListener);

/* Removes a listener from the list of listeners for this Topic. The parameter listenerKey */
/* is used as an index. */
   virtual void unregisterEventListener(const std::string &listenerKey);

/* Returns the /em topicName for this Topic */
   virtual std::string getTopicName();

   void processEvents(const SSIDL::array1<sci::cca::TypeMap::pointer> &eventBodyList);
private:
   friend class EventService;
   //friend class Topic;
/* Should be called only by the EventService    */
   WildcardTopic(const std::string&);
/* Iterates through all the /em listeners for this Topic and calls processEvents() */
/* of each listener for each /em Event */
   
    std::string topicName;
    //SSIDL::array1<sci::cca::TypeMap::pointer> eventBodyList;
    std::map<std::string, sci::cca::IEventListener::pointer> eventListenerMap;
};
}

#endif

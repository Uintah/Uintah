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

#ifndef SCIRun_Internal_Topic_h
#define SCIRun_Internal_Topic_h

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/EventService.h>
#include <SCIRun/SCIRunFramework.h>

namespace SCIRun {


class Topic;
class SCIRunFramework;
typedef SSIDL::array1<sci::cca::Event::pointer> EventPtrList;

/**
 * \class Event
 *
 * An Event contains a header (cca.TypeMap) and a body (cca.TypeMap).
 * (more class desc...)
 *
 */
class Event : public sci::cca::Event {
public:
  Event() {}
  Event(const sci::cca::TypeMap::pointer& theHeader, const sci::cca::TypeMap::pointer& theBody)
     : header(theHeader), body(theBody) {}

  virtual void setHeader(const sci::cca::TypeMap::pointer& h) { header = h; }
  virtual void setBody(const sci::cca::TypeMap::pointer& b) { body = b; }

  virtual sci::cca::TypeMap::pointer getHeader() { return header; }
  virtual sci::cca::TypeMap::pointer getBody() { return body; }

private:
  sci::cca::TypeMap::pointer header;
  sci::cca::TypeMap::pointer body;
};


/**
 * \class Topic
 *
 * A Topic can be considered as a channel for sending events and listening to events.
 * The listeners to \em Topic must implement the EventListener interface
 */

class Topic : public sci::cca::Topic {
public:
  virtual ~Topic();
  //sci::cca::Event::pointer createEvent(const sci::cca::TypeMap::pointer& theHeader, const sci::cca::TypeMap::pointer& theBody);

  /**
   * Sends an Event with the specified \em eventBody.
   * The parameter \em eventBody is a pointer to a CCA TypeMap of the \em message to be sent.
   */
  virtual void sendEvent(const sci::cca::Event::pointer& theEvent);

  /**
   * Adds a \em listener to the collection of listeners for this Topic.
   * The parameter \em listenerKey is used as an index to the collection
   * (STL map) and the parameter \em theListener is the Topic's handle
   * on an EventListener object.
   */
  virtual void registerEventListener(const std::string& listenerKey, const sci::cca::EventListener::pointer& theListener);

  /**
   * Removes a listener from the collection of listeners for this Topic.
   * The parameter \em listenerKey is used as an index.
   */
  virtual void unregisterEventListener(const std::string& listenerKey);

  /**  Returns the \em topicName for this Topic. */
  virtual std::string getTopicName() { return topicName; }

 private:
  friend class EventService;

  // private constructor used only by EventService
  Topic(const std::string& name,SCIRunFramework *fwk);

  /**
   * The following methods should be called only by the EventService.
   *
   * Iterates through all the \em listeners for this Topic and calls processEvents()
   * of each listener for each \em Event.
   */
  void processEvents();

  /**
   * Adds a WildcardTopic to the list of Wildcard Topics that correspond to this Topic.
   * The Parameter \em theWildcardTopic is a pointer to the WildcardTopic that is to be added.
   */
  void addWildcardTopic(const std::string& topicName, const sci::cca::WildcardTopic::pointer& theWildcardTopic);

  /** Removes a WildcardTopic from the list of WildcardTopics. */
  void removeWildcardTopic(const std::string& topicName);

  std::string topicName;
  EventPtrList eventList;
  EventListenerMap eventListenerMap;
  WildcardTopicMap wildcardTopicMap;
  SCIRunFramework *framework;
};


}
#endif

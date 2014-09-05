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

#ifndef Framework_Internal_EventService_h
#define Framework_Internal_EventService_h

#include <sci_metacomponents.h>

#include <Core/CCA/spec/cca_sidl.h>
#include <Framework/Internal/InternalComponentModel.h>
#include <Framework/Internal/InternalFrameworkServiceInstance.h>

#include <map>

namespace SCIRun {

class SCIRunFramework;
class Topic;
class Subscription;

typedef std::map<std::string, sci::cca::Subscription::pointer> SubscriptionMap;
typedef std::map<std::string, sci::cca::Topic::pointer> TopicMap;
typedef std::map<std::string, sci::cca::EventListener::pointer> EventListenerMap;

/**
 * \class EventService
 *
 * The EventService class is used for event communication within the
 * framework or between components.
 * The EventService port is used for event sending and receiving
 * event-related messages.
 *
 * \sa SCIRunFramework
 * \sa Topic
 */

class EventService :  public sci::cca::ports::EventService,
                      public InternalFrameworkServiceInstance {
public:
  virtual ~EventService();

  /**
   * Factory method for creating an instance of a EventService class.
   * Returns a reference counted pointer to a newly-allocated EventService port.
   * The \em framework parameter is a pointer to the relevent framework
   * and the \em name parameter will become the unique name for the new port.
   */
  static InternalFrameworkServiceInstance *create(SCIRunFramework* framework);

  virtual sci::cca::Port::pointer getService(const std::string &) { return sci::cca::Port::pointer(this); }

  /**
   * Creates a new Topic and returns a reference counted pointer  to the newly created
   * instance. If another Topic already exists with  the same \em topicName then,
   * it returns a reference counted pointer to that instance.
   */
  virtual sci::cca::Topic::pointer createTopic(const std::string& topicName);

  /*
   * Creates a new Subscription and returns a reference-counted
   * pointer to the newly created instance.
   */
  virtual sci::cca::Subscription::pointer subscribeToEvents(const std::string& topicName);

  /**
   * Returns a reference-counted pointer to the Topic with the given
   * \em topicName.
   */
  virtual sci::cca::Topic::pointer getTopic(const std::string& topicName);

  /**
   * Returns a reference-counted pointer to the Subscription with the given
   * \em topicName.
   */
  virtual sci::cca::Subscription::pointer getSubscription(const std::string& topicName);

  /** Removes the Topic with this \em topicName from the list of Topics. */
  virtual void releaseTopic(const std::string& topicName);

  /** Removes the Subscription with this \em topicName from the list of Topics. */
  virtual void releaseSubscription(const sci::cca::Subscription::pointer& subscription);

  /** Iterates through the list of topics and calls processEvents() for each Topic. */
  virtual void processEvents();

private:
  EventService(SCIRunFramework* fwk);
  bool isMatch(const std::string& topicName, const std::string& wildcardTopicName);

  TopicMap topicMap;
  SubscriptionMap subscriptionMap;
};

}

#endif

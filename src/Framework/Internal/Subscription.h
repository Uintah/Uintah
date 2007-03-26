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

#ifndef Framework_Subscription_h
#define Framework_Subscription_h

#include <Framework/Internal/EventService.h>
#include <Framework/Internal/Topic.h>
#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {

/**
 * \class Subscription
 *
 * A Subscription is a collection of Topics.
 * One Subscription may correspond to multiple \em Topics.
 * All listeners to Subscriptions must also implement the EventListener interface.
 *
 * \sa Topic
 */
class Subscription : public sci::cca::Subscription {
public:
  virtual ~Subscription();

  /**
   * Adds a \em listener to the collection of listeners for this Subscription.
   * The parameter \em listenerKey is used as an index to the collection
   * (STL map) and the parameter \em theListener is a pointer to the /em Listener class
   */
  virtual void registerEventListener(const std::string &listenerKey, const sci::cca::EventListener::pointer &theListener);

  /**
   * Removes a listener from the collection of listeners for this Topic.
   * The parameter \em listenerKey is used as an index.
   */
  virtual void unregisterEventListener(const std::string &listenerKey);

  /* Returns the \em subscriptionName for this Subscription. */
  virtual std::string getSubscriptionName() { return subscriptionName; }

private:
  friend class EventService;
  friend class Topic;
  const sci::cca::Subscription::pointer subscription;

  // private constructor used only by EventService
  Subscription(const std::string& name);

  /**
   * Iterates through all the \em listeners for this Topic and calls processEvents()
   * on each listener for each \em Event.
   */
  void processEvents(const EventPtrList& eventList);

  std::string subscriptionName;
  EventListenerMap eventListenerMap;
};

}

#endif

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

#ifndef SCIRun_Internal_EventService_h
#define SCIRun_Internal_EventService_h

#include <sci_metacomponents.h>

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>

#include <map>

namespace SCIRun {

class SCIRunFramework;
class Topic;
class WildcardTopic;

typedef std::map<std::string, sci::cca::WildcardTopic::pointer> WildcardTopicMap;
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
 * \sa WildcardTopic
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

  /**
   * Creates an instance of the component of type \em className.  The
   * parameter \em instanceName is the unique name of the newly created
   * instance.
   * Leave \em instanceName empty to have a unique name generated
   * by the framework.
   * This method is implemented through a \em createComponentInstance
   * call to the SCIRunFramework.
   */
  virtual sci::cca::ComponentID::pointer
  createInstance(const std::string& instanceName,
                 const std::string& className,
                 const sci::cca::TypeMap::pointer& properties);

  virtual sci::cca::Port::pointer getService(const std::string &) { return sci::cca::Port::pointer(this); }

  /**
   * Creates a new Topic and returns a reference counted pointer  to the newly created
   * instance. If another Topic already exists with  the same \em topicName then,
   * it returns a reference counted pointer to that instance.
   */
  virtual sci::cca::Topic::pointer createTopic(const std::string& topicName);

  /**
   * Creates a new WildcardTopic and returns a reference-counted pointer
   * to the newly created instance.
   * If another WildcardTopic already exists with the same \em topicName
   * then it returns a reference-counted pointer to that instance.
   */
  virtual sci::cca::WildcardTopic::pointer createWildcardTopic(const std::string& topicName);

  /**
   * Returns a reference-counted pointer to the Topic with the given
   * \em topicName.
   */
  virtual sci::cca::Topic::pointer getTopic(const std::string& topicName);

  /**
   * Returns a reference-counted pointer to the WildcardTopic with the given
   * \em topicName.
   */
  virtual sci::cca::WildcardTopic::pointer getWildcardTopic(const std::string& topicName);

  /** Removes the Topic with this \em topicName from the list of Topics. */
  virtual void releaseTopic(const std::string& topicName);

  /** Removes the WildcardTopic with this \em topicName from the list of Topics. */
  virtual void releaseWildcardTopic(const std::string& topicName);

  /** Iterates through the list of topics and calls processEvents() for each Topic. */
  virtual void processEvents();

private:
  EventService(SCIRunFramework* fwk);
  bool isMatch(const std::string& topicName, const std::string& wildcardTopicName);

  TopicMap topicMap;
  WildcardTopicMap wildcardTopicMap;
};

}

#endif

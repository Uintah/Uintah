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
#ifndef SCIRun_EventService_h
#define SCIRun_EventService_h

#include <sci_metacomponents.h>

#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>
#include<Core/CCA/SSIDL/array.h>
#include <list>
#include <vector>
namespace SCIRun {

class SCIRunFramework;

class EventService :  public sci::cca::ports::EventService,
                      public InternalFrameworkServiceInstance
{
public:
  virtual ~EventService();
  static InternalFrameworkServiceInstance *create(SCIRunFramework* framework);
  virtual sci::cca::ComponentID::pointer
  createInstance(const std::string& instanceName,
		 const std::string& className,
		 const sci::cca::TypeMap::pointer &properties);
  virtual sci::cca::Port::pointer getService(const std::string &);
  
  virtual int createTopic(const std::string &topicName);
  virtual int destroyTopic(const std::string &topicName);
  virtual int sendEvent(const std::string &topicName, const std::string &eventBody);
  //virtual void receiveEvent(const std::string &eventName,std::string &eventBody);
  virtual int registerEventListener(const std::string &topicName,sci::cca::IEventListener::pointer &theListener);
  //virtual void receiveEvent(const std::string &eventName,SSIDL::array1<std::string> &eventBody);
  virtual void processEvents();
private:
  EventService(SCIRunFramework* fwk);
  //std::map<std::string, std::string> eventMap;
  SSIDL::array1<std::string> topicList;
  std::map<std::string, SSIDL::array1<std::string> > eventMap;
  std::map<std::string, SSIDL::array1<sci::cca::IEventListener::pointer> > eventListenerMap;
};
}

#endif

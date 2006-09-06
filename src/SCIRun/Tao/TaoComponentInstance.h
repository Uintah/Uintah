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


/*
 *  TaoComponentInstance.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_TaoComponentInstance_h
#define SCIRun_Framework_TaoComponentInstance_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <SCIRun/Tao/Component.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/Object.h>
#include <Core/Thread/ConditionVariable.h>
#include <map>
#include <string>

namespace SCIRun
{
class TaoPortInstance;
class TaoServices;
class Mutex;

/**
 * \class TaoComponentInstance
 *
 * A handle for an instantiation of a Tao component in the SCIRun framework.
 * This class is a container for information about the Tao component
 * instantiation that is used by the framework.
 */
class TaoComponentInstance : public ComponentInstance,
                             public sci::cca::TaoServices
{
public:
  TaoComponentInstance(SCIRunFramework* framework,
                       const std::string& instanceName,
                       const std::string& className,
                       const sci::cca::TypeMap::pointer& tm,
                       tao::Component* component);
  virtual ~TaoComponentInstance();

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  void registerUsesPort(const std::string& name, const std::string& type,
                        const sci::cca::TypeMap::pointer& properties);

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  void unregisterUsesPort(const std::string& name);

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  void addProvidesPort(const std::string& name,
                       const std::string& type,
                       const sci::cca::TypeMap::pointer& properties);


  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  void removeProvidesPort(const std::string& name);

  // Methods from ComponentInstance
  virtual PortInstance* getPortInstance(const std::string& name);
  virtual PortInstanceIterator* getPorts();

  virtual sci::cca::TypeMap::pointer getPortProperties(const std::string& portName);
  virtual void setPortProperties(const std::string& portName, const sci::cca::TypeMap::pointer& tm);

  sci::cca::ComponentID::pointer getComponentID();

  /** Returns a pointer to the vtk::Component referenced by this class */
  tao::Component* getComponent() {
    return component;
  }

private:
  class Iterator : public PortInstanceIterator
  {
    std::map<std::string, TaoPortInstance*>::iterator iter;
    TaoComponentInstance* comp;
  public:
    Iterator(TaoComponentInstance*);
    virtual ~Iterator();
    virtual PortInstance* get();
    virtual bool done();
    virtual void next();
  private:
    Iterator(const Iterator&);
    Iterator& operator=(const Iterator&);
  };
  std::map<std::string, TaoPortInstance*> ports;
  SCIRun::Mutex lock_ports;

  tao::Component* component;
  Mutex *mutex;

  TaoComponentInstance(const TaoComponentInstance&);
  TaoComponentInstance& operator=(const TaoComponentInstance&);
};

} // end namespace SCIRun

#endif

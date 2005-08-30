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
 *  CCAComponentInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_CCAComponentInstance_h
#define SCIRun_Framework_CCAComponentInstance_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/Object.h>
#include <Core/Thread/ConditionVariable.h>
#include <map>
#include <string>

namespace SCIRun
{
class CCAPortInstance;
class Services;
class Mutex;

/**
 * \class CCAComponentInstance
 *
 * A handle for an instantiation of a CCA component in the SCIRun framework.
 * This class is a container for information about the CCA component
 * instantiation that is used by the framework.
 */
class CCAComponentInstance : public ComponentInstance,
                             public sci::cca::Services
{
public:
  CCAComponentInstance(SCIRunFramework* framework,
                       const std::string& instanceName,
                       const std::string& className,
                       const sci::cca::TypeMap::pointer& typemap,
                       const sci::cca::Component::pointer& component);
  virtual ~CCAComponentInstance();
  
/**
* @param portName The previously registered or provide port which
* 	   the component now wants to use.
* @exception CCAException with the following types: NotConnected, PortNotDefined, 
*                NetworkError, OutOfMemory.
*/
// calls getPortNonblocking from inside critical section
sci::cca::Port::pointer getPort(const std::string& name);

/**
* @return The named port, if it exists and is connected or self-provided,
* 	      or NULL if it is registered and is not yet connected. Does not
* 	      return if the Port is neither registered nor provided, but rather
* 	      throws an exception.
* @param portName registered or provided port that
* 	     the component now wants to use.
* @exception CCAException with the following types: PortNotDefined, OutOfMemory.
*/
// throws CCA exception if port type is PROVIDES (??)
// returns null if port's connections vector size != 1
// otherwise returns port peer (1st element of connections vector)
  sci::cca::Port::pointer getPortNonblocking(const std::string& name);

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */ 
  void releasePort(const std::string& name);

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  sci::cca::TypeMap::pointer createTypeMap();

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  void registerUsesPort(const std::string& name, const std::string& type,
                        const sci::cca::TypeMap::pointer& properties);

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  void unregisterUsesPort(const std::string& name);

  /** A proxy method for gov::cca::Services.  Calls the corresponding method in
      SCIRunFramework::Services. */
  void addProvidesPort(const sci::cca::Port::pointer& port,
                       const std::string& name,
                       const std::string& type,
                       const sci::cca::TypeMap::pointer& properties);

  /** A proxy method for gov::cca::Services.  Calls the corresponding
      method in SCIRunFramework::Services. */
  void removeProvidesPort(const std::string& name);

  /** Returns the complete list of the properties for a Port.
   *
   * Keys:
   * cca.portName
   * cca.portType
   * Properties from addProvidesPort/registerUsesPort...
   */
  sci::cca::TypeMap::pointer getPortProperties(const std::string& portName);

  /** A proxy method for gov::cca::Services.  Calls the corresponding
      method in SCIRunFramework::Services. */
  sci::cca::ComponentID::pointer getComponentID();
  
  // Methods from ComponentInstance
  virtual PortInstance* getPortInstance(const std::string& name);
  virtual PortInstanceIterator* getPorts();
  virtual void registerForRelease(const sci::cca::ComponentRelease::pointer &compRel);

private:
  class Iterator : public PortInstanceIterator
  {
    std::map<std::string, CCAPortInstance*>::iterator iter;
    CCAComponentInstance* comp;
  public:
    Iterator(CCAComponentInstance*);
    virtual ~Iterator();
    virtual PortInstance* get();
    virtual bool done();
    virtual void next();
  private:
    Iterator(const Iterator&);
    Iterator& operator=(const Iterator&);
  };
  std::map<std::string, CCAPortInstance*> ports;
  SCIRun::Mutex lock_ports;

  std::map<std::string, std::vector<Object::pointer> > preports;
  SCIRun::Mutex lock_preports;

  std::map<std::string, int > precnt;
  std::map<std::string, ConditionVariable*> precond;
  
  sci::cca::Component::pointer component;
  Mutex *mutex;
  int size;
  int rank;
  
  CCAComponentInstance(const CCAComponentInstance&);
  CCAComponentInstance& operator=(const CCAComponentInstance&);
};

} // end namespace SCIRun

#endif

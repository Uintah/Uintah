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
 *  CorbaComponentInstance.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Corba_CorbaComponentInstance_h
#define SCIRun_Corba_CorbaComponentInstance_h

#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <SCIRun/Corba/Port.h>
#include <SCIRun/Corba/Component.h>
#include <SCIRun/Corba/CorbaPortInstance.h>
#include <string>
#include <vector>

namespace SCIRun {

class CCAPortInstance;
class Module;

/**
 * \class CorbaComponentInstance
 *
 * A handle to an instantiation of a Corba component in the SCIRun framework.  This
 * class is a container for information necessary to interact with an
 * instantiation of a Corba component such as its unique instance name, its
 * type, its ports, and the framework to which it belongs.
 *
 */
class CorbaComponentInstance : public ComponentInstance {
  friend class CorbaPortInstance;
public:
  /** Constructor specifies the SCIRunFramework to which this instance belongs
      (\em fwk), the unique \em instanceName of the component instance, and a
      pointer to an allocated corba::Component object.*/
  CorbaComponentInstance(SCIRunFramework* fwk,
                       const std::string& instanceName,
                       const std::string& className,
                       corba::Component * component);
  virtual ~CorbaComponentInstance();
  
  // Methods from ComponentInstance
  /** Returns a pointer to the port named \em name.  If no such port exists in
      this component, returns a null pointer. */
  virtual PortInstance* getPortInstance(const std::string& name);
  /** Returns the list of ports associated with this component. */
  virtual PortInstanceIterator* getPorts();

  /** Returns a pointer to the corba::Component referenced by this class */
  corba::Component* getComponent() {
    return component;
    }
  private:
    class Iterator : public PortInstanceIterator {
    public:
      Iterator(CorbaComponentInstance*);
      virtual ~Iterator();
      virtual PortInstance* get();
      virtual bool done();
      virtual void next();
    private:
      Iterator(const Iterator&);
      Iterator& operator=(const Iterator&);

      CorbaComponentInstance* ci;
      int index;
    };
    corba::Component* component;
    std::vector<CCAPortInstance*> specialPorts;
    CorbaComponentInstance(const CorbaComponentInstance&);
    CorbaComponentInstance& operator=(const CorbaComponentInstance);
  };
}

#endif

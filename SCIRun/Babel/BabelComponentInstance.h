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
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */
#ifndef SCIRun_Framework_BabelComponentInstance_h
#define SCIRun_Framework_BabelComponentInstance_h

#include <SCIRun/Babel/BabelPortInstance.h>
#include <SCIRun/ComponentInstance.h>
#include <SCIRun/PortInstanceIterator.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <SCIRun/Babel/framework.hh>
#include <SCIRun/Babel/gov_cca.hh>
#include <map>
#include <string>

namespace SCIRun {

  class BabelCCAGoPort : public virtual sci::cca::ports::GoPort{
  public:
    BabelCCAGoPort(const gov::cca::ports::GoPort &port);
    virtual int go();
  private:
    gov::cca::ports::GoPort port;
  };

  class BabelCCAUIPort : public virtual sci::cca::ports::UIPort{
  public:
    BabelCCAUIPort(const gov::cca::ports::UIPort &port);
    virtual int ui();
  private:
    gov::cca::ports::UIPort port;
  };


  class BabelPortInstance;
  class Services;

  class BabelComponentInstance : public ComponentInstance{
  public:
    BabelComponentInstance(SCIRunFramework* framework,
			 const std::string& instanceName,
			 const std::string& className,
			 const gov::cca::TypeMap& typemap,
			 const gov::cca::Component& component,
			 const framework::Services& svc);
    virtual ~BabelComponentInstance();

    // Methods from gov::cca::Services
    gov::cca::Port getPort(const std::string& name);
    gov::cca::Port getPortNonblocking(const std::string& name);
    void releasePort(const std::string& name);
    gov::cca::TypeMap createTypeMap();
    void registerUsesPort(const std::string& name, const std::string& type,
			  const gov::cca::TypeMap& properties);
    void unregisterUsesPort(const std::string& name);
    void addProvidesPort(const gov::cca::Port& port,
			 const std::string& name,
			 const std::string& type,
			 const gov::cca::TypeMap& properties);
    void removeProvidesPort(const std::string& name);
    gov::cca::TypeMap getPortProperties(const std::string& portName);
    gov::cca::ComponentID getComponentID();

    // Methods from ComponentInstance
    virtual PortInstance* getPortInstance(const std::string& name);
    virtual PortInstanceIterator* getPorts();
 private:
    framework::Services svc;
    class Iterator : public PortInstanceIterator {
      std::map<std::string, PortInstance*> *ports;
      std::map<std::string, PortInstance*>::iterator iter;
    public:
      Iterator(BabelComponentInstance*);
      virtual ~Iterator();
      virtual PortInstance* get();
      virtual bool done();
      virtual void next();
    private:
      Iterator(const Iterator&);
      Iterator& operator=(const Iterator&);
      //sci::cca::ComponentID::pointer cid;
    };

    gov::cca::Component component;
    BabelComponentInstance(const BabelComponentInstance&);
    BabelComponentInstance& operator=(const BabelComponentInstance&);
  };
}

#endif





















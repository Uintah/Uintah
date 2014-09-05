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
 *  ComponentInstance.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_ComponentInstance_h
#define SCIRun_Framework_ComponentInstance_h

#include <SCIRun/CCA/ComponentID.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <string>

namespace SCIRun
{

class SCIRunFramework;
class PortInstance;
class PortInstanceIterator;

/**
 * \class ComponentInstance
 *
 * A handle to an instantiation of a component in the SCIRun framework.  This
 * class is a container for information necessary to interact with an
 * instantiation of a framework component such as its unique instance name, its
 * type, its ports, and the framework to which it belongs.  Specific component
 * models may subclass ComponentInstance to define their own variations. */
class ComponentInstance
{
public:
    ComponentInstance(SCIRunFramework* framework,
                      const std::string& instanceName,
                      const std::string& className,
                      const sci::cca::TypeMap::pointer& tm);
    virtual ~ComponentInstance();

    /** The framework to which this component instance belongs, i.e. the
        framework in which it was instantiated. */
    SCIRunFramework* framework;

    /** Returns a pointer to the port named \em name.  If no such port exists in
        this component, returns a null pointer. */
    virtual PortInstance* getPortInstance(const std::string& name) = 0;

    /** Returns the list of ports associated with this component. */
    virtual PortInstanceIterator* getPorts() = 0;

    inline void
    setInstanceName(const std::string &name) { instanceName = name; }

    inline std::string
    getInstanceName() const { return instanceName; }

    inline std::string
    getClassName() const { return className; }
        
    inline sci::cca::TypeMap::pointer&
    getComponentProperties() { return comProperties; }
        
    void
    setComponentProperties(const sci::cca::TypeMap::pointer &tm);

    bool
    releaseComponentCallback(const sci::cca::Services::pointer &svc);

protected:
  /** The unique name of this component instance. */
  std::string instanceName;

  /** The type of the component. */
  std::string className;

  sci::cca::TypeMap::pointer comProperties;

  /** See interface ComponentRelease */
  sci::cca::ComponentRelease::pointer releaseCallback;

private:
  ComponentInstance(const ComponentInstance&);
  ComponentInstance& operator=(const ComponentInstance&);
};

} // end namespace SCIRun

#endif

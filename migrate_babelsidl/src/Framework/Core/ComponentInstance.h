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

#ifndef Framework_Framework_ComponentInstance_h
#define Framework_Framework_ComponentInstance_h

#include <Framework/sidl/Impl/glue/scijump.hxx>
#include <Framework/Core/PortInstance.h>
#include <string>
#include <map>

using namespace scijump;

namespace SCIRun
{

class SCIRunFramework;
class PortInstance;
class PortInstanceIterator;

typedef std::map<std::string, PortInstance*> PortInstanceMap;

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
  ComponentInstance(SCIJumpFramework* framework,
		    const std::string& instanceName,
		    const std::string& className,
		    const gov::cca::TypeMap& tm);
  virtual ~ComponentInstance();

  /** The framework to which this component instance belongs, i.e. the
      framework in which it was instantiated. */
  SCIJumpFramework* framework;

  /** Returns a pointer to the port named \em name.  If no such port exists in
      this component, returns a null pointer. */
  virtual PortInstance* getPortInstance(const std::string& name) = 0;

  /** Returns the list of ports associated with this component. */
  virtual PortInstanceIterator* getPorts() = 0;

  /** Returns the complete list of the properties for a Port.
   * These may include properties set when the port is registered and
   * properties set by the framework.
   * Properties will include the following:
   * <pre>
   *     key             standard values
   * cca.portName      port registration name (string)
   * cca.portType      port registration type (string)
   * </pre>
   */
  virtual gov::cca::TypeMap getPortProperties(const std::string& portName) = 0;

  /** Sets the port properties associated with this component. */
  virtual void setPortProperties(const std::string&, const gov::cca::TypeMap&) = 0;

  inline void
  setInstanceName(const std::string &name) { instanceName = name; }

  inline std::string
  getInstanceName() const { return instanceName; }

  inline std::string
  getClassName() const { return className; }

  inline gov::cca::TypeMap&
  getComponentProperties() { return properties; }

  void
  setComponentProperties(const gov::cca::TypeMap &tm);

  bool
  releaseComponentCallback(const gov::cca::Services &svc);

protected:
  /** The unique name of this component instance. */
  std::string instanceName;

  /** The type of the component. */
  std::string className;

  gov::cca::TypeMap properties;

  /** See interface ComponentRelease */
  gov::cca::ComponentRelease releaseCallback;

private:
  ComponentInstance(const ComponentInstance&);
  ComponentInstance& operator=(const ComponentInstance&);
};

} // end namespace SCIRun

#endif

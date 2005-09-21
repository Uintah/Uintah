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
 *  ComponentInfoImpl.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Distributed_ComponentInfoImpl_h
#define SCIRun_Distributed_ComponentInfoImpl_h

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/CCA/spec/sci_sidl.h>
#include <map>
#include <string>

namespace SCIRun {
  
  class DistributedFramework;
  namespace Distributed = sci::cca::distributed;

  /**
   * \class ComponentInfo
   *
   */
  
  template<class Base>
  class ComponentInfoImpl : public Base
  {
  private:
    typedef std::map<std::string, Distributed::PortInfo::pointer> PortMap;
    
  public:
    typedef Distributed::ComponentInfo::pointer pointer;
    
    
    ComponentInfoImpl(const Distributed::DistributedFramework::pointer &framework,
		      const std::string& instanceName,
		      const std::string& className,
		      const sci::cca::TypeMap::pointer& typemap,
		      const sci::cca::Component::pointer& component);
    virtual ~ComponentInfoImpl();
    
    /*
     * sci.cca.distributed.ComponentInfo interface
     */
    
    virtual Distributed::DistributedFramework::pointer getFramework();
    virtual SSIDL::array1<Distributed::PortInfo::pointer> getPorts();
    virtual Distributed::PortInfo::pointer getPortInfo(const std::string& name);
    virtual std::string getClassName();
    virtual sci::cca::TypeMap::pointer getProperties();
    virtual void setProperties(const sci::cca::TypeMap::pointer &);
    
    /*
     * cca::ComponentID interface
     */
    virtual std::string getInstanceName();
    virtual std::string getSerialization();
    
  private:
    PortMap ports;
    
    sci::cca::Component::pointer component;
    Mutex *mutex;
    
    std::string className;
    sci::cca::TypeMap::pointer properties;
    
    // from ComponentID
    Distributed::DistributedFramework::pointer framework;
    std::string instanceName;
    
    // prevent using these directly
    ComponentInfoImpl(const ComponentInfoImpl&);
    ComponentInfoImpl& operator=(const ComponentInfoImpl&);
  };
  
} // end namespace SCIRun

#include <SCIRun/Distributed/ComponentInfoImpl.code>

#endif

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
 *  CoreServicesImpl.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Core_CoreServicesBase_h
#define SCIRun_Core_CoreServicesBase_h

#include <SCIRun/Core/ComponentInfoBase.h>


namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::core;


  /**
   * \class CoreServices
   *
   */
  
  template<class Interface>
  class CoreServicesBase : public ComponentInfoBase<Interface>
  {
  public:
    typedef CoreServices::pointer pointer;
    typedef typename ComponentInfoBase<Interface>::PortMap PortMap;

    using ComponentInfoBase<Interface>::ports_lock;
    using ComponentInfoBase<Interface>::componentPorts;

    //using typename ComponentInfoBase<Interface>::ports_lock;
    //using typename ComponentInfoBase<Interface>::ports;

    CoreServicesBase(const CoreFramework::pointer &framework,
		     const std::string& instanceName,
		     const std::string& className,
		     const TypeMap::pointer& properties,
		     const Component::pointer& component);

    virtual ~CoreServicesBase();

    // from cca.Services
    virtual void releasePort(const std::string&);
    virtual void registerUsesPort(const std::string&, const std::string&, const TypeMap::pointer &);
    virtual void unregisterUsesPort(const std::string&);
    virtual void addProvidesPort(const Port::pointer &, 
				 const std::string&, const std::string&, const TypeMap::pointer &);
    virtual void registerForRelease( const ComponentRelease::pointer &);
    virtual void removeProvidesPort(const std::string&);

    virtual Port::pointer getPort(const std::string&);
    virtual Port::pointer getPortNonblocking(const std::string&);
    virtual TypeMap::pointer createTypeMap();
    virtual TypeMap::pointer getPortProperties(const std::string &);
    virtual ComponentID::pointer getComponentID();

  protected:
    typedef std::map<std::string, ServiceInfo::pointer> ServicePortMap;
    ServicePortMap servicePorts;
    Mutex service_lock;

    // prevent using these directly
    CoreServicesBase<Interface>(const CoreServicesBase<Interface>&);
    CoreServicesBase<Interface>& operator=(const CoreServicesBase<Interface>&);
  };
  
} // end namespace SCIRun

#endif

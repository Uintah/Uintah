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
 *  ServiceRegistryPortBase.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Core_ServiceRegistryPortBase_h
#define SCIRun_Core_ServiceRegistryPortBase_h

#include <SCIRun/Core/ServiceRegistryCoordinator.h>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::ports;
  using namespace sci::cca::core;

  /**
   * \class ServiceRegistryPortBase
   *
   */
  
  template<class Interface>
  class ServiceRegistryPortBase : public Interface
  {
  public:
    typedef sci::cca::ports::ServiceRegistry::pointer pointer;
    
    ServiceRegistryPortBase(ServiceRegistryCoordinator *registry, const ComponentInfo::pointer &requester )
      : registry(registry), requester(requester)
    {}

    virtual ~ServiceRegistryPortBase() {}
    
    /*
     * sci.cca.core.ServiceRegistry interface
     */
    
    virtual bool addService(const std::string &serviceType, const ServiceProvider::pointer &portProvider)
    {
      return registry->addService( serviceType, portProvider, requester );
    }

    virtual bool addSingletonService( const std::string &serviceType, const Port::pointer &server) 
    {
      return registry->addSingletonService( serviceType, server, requester );
    }

    virtual void removeService( const std::string &serviceType )
    {
      registry->removeService( serviceType, requester );
    }
      
  protected:
    ServiceRegistryCoordinator *registry;
    ComponentInfo::pointer requester;

    // prevent using these directly
    ServiceRegistryPortBase(const ServiceRegistryPortBase&);
    ServiceRegistryPortBase& operator=(const ServiceRegistryPortBase&);
  };
  
} // end namespace SCIRun

#endif

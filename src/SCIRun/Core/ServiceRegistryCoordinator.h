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
 *  ServiceRegistryCoordinator.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Core_ServiceRegistryCoordinator_h
#define SCIRun_Core_ServiceRegistryCoordinator_h

#include <Core/Thread/Mutex.h>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::ports;
  using namespace sci::cca::core;
  
  /**
   * \class ServiceRegistryCoordinatorBase
   *
   */
  
  class ServiceRegistryCoordinator
  {
  public:
    typedef ServiceRegistryCoordinator * pointer;
    
    ServiceRegistryCoordinator(const CoreFramework::pointer &framework);
    virtual ~ServiceRegistryCoordinator();
    
    /*
     * sci.cca.core.ServiceRegistryCoordinator interface
     */
    
    virtual bool addService(const std::string &serviceType, 
			    const ServiceProvider::pointer &portProvider,
			    const ComponentInfo::pointer &requester);
    virtual bool addSingletonService( const std::string &serviceType, const Port::pointer &server, const ComponentInfo::pointer &requester);
    virtual void removeService( const std::string &serviceType, const ComponentInfo::pointer &requester);

  protected:
    CoreFramework::pointer framework;
    
    typedef std::map<std::string, ComponentInfo::pointer> ProvidedMap;
    ProvidedMap providedServices;

    Mutex lock;
    // prevent using these directly
    ServiceRegistryCoordinator(const ServiceRegistryCoordinator&);
    ServiceRegistryCoordinator& operator=(const ServiceRegistryCoordinator&);
  };
  
} // end namespace SCIRun

#endif

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
 *  ServiceRegistryCoordinator.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#include <Core/CCA/spec/sci_sidl.h>
#include <Core/Thread/Guard.h>
#include <SCIRun/Core/CCAException.h>
#include <SCIRun/Core/ServiceRegistryCoordinator.h>
#include <SCIRun/Core/ProviderServiceFactoryImpl.h>
#include <SCIRun/Core/FixedPortServiceFactoryImpl.h>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::core;
  
  /**
   * \class ServiceRegistryCoordinator
   *
   */
  
  ServiceRegistryCoordinator::ServiceRegistryCoordinator(const CoreFramework::pointer &framework )
    : framework(framework), lock("ServiceRegistryCoordinator lock")
  {}
  
  ServiceRegistryCoordinator::~ServiceRegistryCoordinator()
  {}

  
  bool
  ServiceRegistryCoordinator::addService(const std::string &serviceType, 
			      const ServiceProvider::pointer &portProvider, 
			      const ComponentInfo::pointer &requester)
  {
    Guard guard(&lock);

    if ( framework->isFrameworkService(serviceType) )
      return false;

    ServiceFactory::pointer factory = new ProviderServiceFactoryImpl( framework, serviceType, portProvider, requester);
    if ( !framework->addFrameworkServiceFactory(factory) ) 
      return false;

    providedServices[serviceType] = requester;
    
    return true;
  }

  bool
  ServiceRegistryCoordinator::addSingletonService( const std::string &serviceType, 
					const Port::pointer &server,
					const ComponentInfo::pointer &requester)
  {
    Guard guard(&lock);

    ServiceFactory::pointer factory =  new FixedPortServiceFactoryImpl( framework, serviceType, server);
    if (!framework->addFrameworkServiceFactory(factory) )
      return false;

    providedServices[serviceType] = requester;
    return true;
  }
  
  void
  ServiceRegistryCoordinator::removeService( const std::string &serviceType, const ComponentInfo::pointer &requester)
  {
    Guard guard(&lock);

    ProvidedMap::iterator iter = providedServices.find(serviceType);
    if ( iter == providedServices.end() ) {
      if ( framework->isFrameworkService(serviceType) )
	throw CCAException::create("Can not remove service [" +serviceType+"] as it is a Framework service");
      else
	throw CCAException::create("Can not remove provided service. unknown services [" +serviceType+"]");
    }
    if ( iter->second != requester )
	throw CCAException::create("Can not remove provided service [" +serviceType+"]. service provided by a different component");

    framework->removeFrameworkServiceFactory( serviceType );
    providedServices.erase(iter);
  }
} // end namespace SCIRun


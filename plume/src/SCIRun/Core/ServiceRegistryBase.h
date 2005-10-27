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
 *  ServiceRegistryBase.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */

#ifndef SCIRun_Core_ServiceRegistryBase_h
#define SCIRun_Core_ServiceRegistryBase_h

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::ports;
  using namespace sci::cca::core;
  
  /**
   * \class ServiceRegistryBase
   *
   */
  
  template<class Interface>
  class ServiceRegistryBase : public Interface
  {
  public:
    typedef ServiceRegistry::pointer pointer;
    
    ServiceRegistryBase(const CoreFramework::pointer &framework, const ComponentInfo::pointer &requester );
    virtual ~ServiceRegistryBase();
    
    /*
     * sci.cca.core.ServiceRegistry interface
     */
    
    virtual bool addService(const std::string &serviceType, const ServiceProvider::pointer &portProvider);
    virtual bool addSingletonService( const std::string &serviceType, const Port::pointer &server);

  protected:
    CoreFramework::pointer framework;
    ComponentInfo::pointer requester;

    // prevent using these directly
    ServiceRegistryBase(const ServiceRegistryBase&);
    ServiceRegistryBase& operator=(const ServiceRegistryBase&);
  };
  
} // end namespace SCIRun

#endif

/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2005 Scientific Computing and Imaging Institute,
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
 *  CoreFrameworkBase.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#ifndef SCIRun_Framework_CoreFrameworkBase_h
#define SCIRun_Framework_CoreFrameworkBase_h

#include <list>

namespace SCIRun {

  using namespace sci::cca;
  using namespace sci::cca::core;

  /**
   * \class CoreFramework
   * 
   * \brief An implementation of a CoreFramework 
   */


  template<class Interface>
  class CoreFrameworkBase : public Interface
  {
  public:
    typedef CoreFramework::pointer pointer;

    CoreFrameworkBase();
    virtual ~CoreFrameworkBase();

    virtual SSIDL::array1<ComponentClassDescription::pointer> listAllComponentTypes( bool );

    /*
     * methods to implement AbstractFractory
     */

    virtual TypeMap::pointer createTypeMap();
    
    virtual Services::pointer getServices( const std::string &name, 
					   const std::string &type, 
					   const TypeMap::pointer &pointer);
    
    virtual void releaseServices( const Services::pointer &services);
    
    virtual void shutdownFramework();

    /*
     * methods that implement the CoreFramework 
     */

    // info
    virtual ComponentID::pointer lookupComponentID(const std::string &);
    virtual SSIDL::array1<ComponentID::pointer> getComponentIDs();
    virtual SSIDL::array1<ConnectionID::pointer> getConnectionIDs(const SSIDL::array1<ComponentID::pointer> &);

    // component manipulations
    virtual ComponentInfo::pointer  createInstance(const std::string &, const std::string &, const TypeMap::pointer&) ;
    virtual void destroyInstance(const ComponentID::pointer &id);

    // connections
    virtual void addConnection(const ConnectionID::pointer &connection) ;
    virtual void disconnect(const ConnectionID::pointer &connection);

    // services
    virtual bool isFrameworkService(const std::string &);
    virtual ServiceInfo::pointer getFrameworkService(const std::string &, 
						     const PortInfo::pointer &, 
						     const ComponentInfo::pointer &);
    virtual void releaseFrameworkService(const ServiceInfo::pointer &);

    // factories
    virtual void addComponentClassFactory( const ComponentClassFactory::pointer &factory );
    virtual bool addFrameworkServiceFactory( const ServiceFactory::pointer &factory );

  private:
    typedef std::list<ConnectionInfo::pointer> ConnectionList;
    typedef std::map<std::string, ComponentInfo::pointer> ComponentMap;
    typedef std::map<std::string, ComponentClassFactory::pointer> ComponentClassFactoryMap;
    typedef std::map<std::string, ServiceFactory::pointer> ServiceMap;

    ConnectionList connections;
    ComponentMap  components;
    ServiceMap     services;
    ComponentClassFactoryMap factories;

    Mutex connection_lock;
    Mutex component_lock;
    Mutex factory_lock;
    Mutex service_lock;
  };
  
} // end namespace SCIRun

//#include <SCIRun/Distributed/CoreFramework.code>

#endif

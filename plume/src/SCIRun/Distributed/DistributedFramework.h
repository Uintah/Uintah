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
 *  DistributedFramework.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#ifndef SCIRun_Framework_DistributedFramework_h
#define SCIRun_Framework_DistributedFramework_h

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Distributed/DistributedFrameworkImpl.h>
#include <SCIRun/Distributed/ComponentDescription.h>
#include <list>

namespace SCIRun {

  namespace Distributed = sci::cca::distributed;

  /**
   * \class DistributedFramework
   * 
   * \brief An implementation of a DistributedFramework 
   */
  
  class DistributedFramework : public DistributedFrameworkImpl<Distributed::DistributedFramework>
  {
  public:
    typedef DistributedFrameworkImpl<Distributed::DistributedFramework>::pointer pointer;
    typedef Distributed::internal::Service::pointer ServicePointer;

    DistributedFramework( pointer parent = 0);
    virtual ~DistributedFramework();

    /*
     * Two pure virtual methods to create and destroy a component.
     * These methods must be defined in the Framework that derives from this DistributedFramework base
     */
    virtual sci::cca::Component::pointer 
    createComponent( const std::string &, const std::string &, const sci::cca::TypeMap::pointer& properties) = 0;

    virtual void destroyComponent( const sci::cca::ComponentID::pointer &) = 0;

    /*
     * methods that implement the DistributedFramework 
     */

    virtual Distributed::ComponentInfo::pointer 
    createInstance( const std::string &, const std::string &, const sci::cca::TypeMap::pointer& properties) ;

    
    void listAllComponentTypes(std::vector<ComponentDescription*> &, bool );
    sci::cca::ComponentID::pointer createComponentInfo(const std::string &, 
						       const std::string &, 
						       const sci::cca::TypeMap::pointer &);

    sci::cca::ComponentID::pointer lookupComponentID(const std::string &);

    SSIDL::array1<sci::cca::ComponentID::pointer> getComponentIDs();
    SSIDL::array1<sci::cca::ConnectionID::pointer> getConnectionIDs(const SSIDL::array1<sci::cca::ComponentID::pointer> &componentList);

    void addConnection(sci::cca::ConnectionID::pointer);
    void disconnect(const sci::cca::ConnectionID::pointer& connection);

    void destroyComponentInfo(const sci::cca::ComponentID::pointer &id);

    ServicePointer getFrameworkService(const std::string &);
    void releaseFrameworkService(const ServicePointer &service);

  private:
    typedef std::list<Distributed::ConnectionInfo::pointer> ConnectionList;
    typedef std::list<Distributed::ComponentInfo::pointer> ComponentList;
    typedef std::map<std::string, Distributed::internal::Service::pointer> ServiceMap;

    ConnectionList connections;
    ComponentList  components;
    ServiceMap     services;

    Mutex connection_lock;
    Mutex component_lock;
  };
  
} // end namespace SCIRun

#endif

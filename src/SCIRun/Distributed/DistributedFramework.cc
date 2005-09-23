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
 *  DistributedFramework.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#include <SCIRun/Distributed/DistributedFramework.h>
#include <SCIRun/Distributed/ConnectionID.h>
#include <SCIRun/Distributed/ConnectionInfo.h>
#include <SCIRun/Distributed/ComponentID.h>
#include <SCIRun/Distributed/ComponentInfo.h>
#include <Core/Thread/Guard.h>
#include <algorithm>

namespace SCIRun {
  
  DistributedFramework::DistributedFramework(pointer parent )
    : DistributedFrameworkImpl<Distributed::DistributedFramework>(parent),
      connection_lock("DistributedFramework::connection_lock"),
      component_lock("DistributedFramework::component_lock")
  {
  }
  
  DistributedFramework::~DistributedFramework()
  {
  }
  

  ComponentInfo::pointer DistributedFramework::createInstance( const std::string& instanceName,
							       const std::string& className,
							       const sci::cca::TypeMap::pointer& properties)
  {
    // call the virtual method that actually create a component
    return ComponentInfo::pointer( createComponent( instanceName, className, properties) );
  }

  ComponentID::pointer DistributedFramework::lookupComponentID(const std::string &name) 
  {
    SCIRun::Guard guard(&component_lock);

    for (ComponentList::const_iterator c = components.begin(); c != components.end(); ++c)
      if ( (*c)->getInstanceName() == name )
	return (*c);
    return ComponentID::pointer(0);
  }

  void DistributedFramework::listAllComponentTypes(std::vector<ComponentDescription*> &, bool )
  {
    // FIXME: [yarden] need to copy this function
  }

  SSIDL::array1<ConnectionID::pointer>
  DistributedFramework::getConnectionIDs(const SSIDL::array1<ComponentID::pointer> &componentList)
  {
    SSIDL::array1<ConnectionID::pointer> selected;

    SCIRun::Guard guard(&connection_lock);

    for (ConnectionList::const_iterator iter = connections.begin(); iter != connections.end(); ++iter ) {
      const ConnectionID::pointer &connection = (*iter);

      ComponentID::pointer user = connection->getUser();
      ComponentID::pointer provider = connection->getProvider();

      for (unsigned j = 0; j < componentList.size(); j++) {
	const ComponentID::pointer &component = componentList[j];
	if (user == component || provider == component) {
	  selected.push_back(connection);
	  break;
	}
      }
    }
    return selected;
  }

  
  void DistributedFramework::disconnect(const ConnectionID::pointer &connection)
  {
    SCIRun::Guard guard(&connection_lock);
    
    ConnectionList::iterator c = find( connections.begin(), connections.end(), pidl_cast<ConnectionInfo::pointer>(connection));
    if ( c != connections.end() )
      connections.erase(c);
    // TODO [yarden]: 
    // else throw exception
  }


  void DistributedFramework::destroyComponentInfo( const ComponentID::pointer &component )
  {
    SCIRun::Guard guard(&component_lock);
    
    // call the virtual function that actually delete the component 
    ComponentInfo::pointer info = pidl_cast<ComponentInfo::pointer>(component);
    destroyComponent(info);
    
    ComponentList::iterator c = find( components.begin(), components.end(), pidl_cast<ComponentInfo::pointer>(component));
    if ( c != components.end() ) {
      // erase the last reference to the component info
      components.erase(c);
    }
    else {
      // TODO [yarden]: 
      // throw exception ?
    }
  }

  DistributedFramework::ServicePointer
  DistributedFramework::getFrameworkService(const std::string &name)
  {
    ServiceMap::const_iterator service = services.find(name);
    return service != services.end() ? (*service).second : ServicePointer(0);
  }

  void
  DistributedFramework::releaseFrameworkService(const ServicePointer &service)
  {
    // for now we only have singleton services and we can forgo deleting them.
  }


} // SCIRun namespace

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
 *  DistributedFramework.cc: An instance of the SCIRun framework
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

namespace SCIRun {
  
  DistributedFramework::DistributedFramework(pointer parent )
    : DistributedFrameworkImpl<Distributed::DistributedFramework>(parent)
  {
  }
  
  DistributedFramework::~DistributedFramework()
  {
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

    for (unsigned i = 0; i < connections.size(); i++) {
      ConnectionID::pointer &connection = connections[i];
      ComponentID::pointer user = connection->getUser();
      ComponentID::pointer provider = connections->getProvider();
      for (unsigned j = 0; j < componentList.size(); j++) {
	const ComponentID::pointer &component = componentList[j];
	if (user == component || proviver == component) {
	  selected.push_back(connection]);
	break;
      }
    }
    return selected;
  }

  
  // TODO: timeout never used
  void DistribugedFramework::disconnect(const ConnectionID::pointer& connection, float/* timeout*/)
  {
    ComponentInfo::pointer user = connnetion->getUser();
    ComponentInfo::pointer provider = connection->getProvider();
    
    PortInfo::pointer userPort = user->getPortInfo(connection->getUserPortName());
    PortInfo::pointer providerPort = provider->getPortInfo(connection->getProviderPortName());
    
    SCIRun::Guard guard(&connection_lock);
    
    userPort->disconnect(providerPort);
    
    ConnectionList::iterator c = connections.find(connection);
    if ( c != connections.end() )
      connections.erase(c);
    // TODO [yarden]: 
    // else throw exception
    
    //std::cerr << "BuilderService::disconnect: timeout or safty check needed "<<std::endl;
  }

} // SCIRun namespace

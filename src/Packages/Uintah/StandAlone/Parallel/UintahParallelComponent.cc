
#include <Uintah/Parallel/UintahParallelCore/CCA/Component.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <algorithm>

using namespace Uintah;
using namespace std;

UintahParallelCore/CCA/Component::UintahParallelCore/CCA/Component(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

UintahParallelCore/CCA/Component::~UintahParallelCore/CCA/Component()
{
  for(map<string, PortRecord*>::iterator iter = portmap.begin(); 
      iter != portmap.end(); iter++) 
    delete iter->second;

}

void
UintahParallelCore/CCA/Component::attachPort(const string& name,
				    Packages/UintahParallelPort* port)
{
    map<string, PortRecord*>::iterator iter = portmap.find(name);
    if(iter == portmap.end()){
	portmap[name]=scinew PortRecord(port);
    } else {
	iter->second->connections.push_back(port);
    }
}

UintahParallelCore/CCA/Component::PortRecord::PortRecord(UintahParallelPort* port)
{
    connections.push_back(port);
}

UintahParallelPort* UintahParallelCore/CCA/Component::getPort(const std::string& name)
{
    map<string, PortRecord*>::iterator iter = portmap.find(name);
    if(iter == portmap.end())
	return 0;
    else if(iter->second->connections.size()> 1)
	return 0;
    else
	return iter->second->connections[0];
}

void UintahParallelCore/CCA/Component::releasePort(const std::string&)
{
}


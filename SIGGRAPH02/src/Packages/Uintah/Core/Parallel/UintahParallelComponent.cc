
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <algorithm>

using namespace Uintah;
using namespace std;

UintahParallelComponent::UintahParallelComponent(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

UintahParallelComponent::~UintahParallelComponent()
{
  for(map<string, PortRecord*>::iterator iter = portmap.begin(); 
      iter != portmap.end(); iter++) 
    delete iter->second;

}

void
UintahParallelComponent::attachPort(const string& name,
				    UintahParallelPort* port)
{
    map<string, PortRecord*>::iterator iter = portmap.find(name);
    if(iter == portmap.end()){
	portmap[name]=scinew PortRecord(port);
    } else {
	iter->second->connections.push_back(port);
    }
}

UintahParallelComponent::PortRecord::PortRecord(UintahParallelPort* port)
{
    connections.push_back(port);
}

UintahParallelPort* UintahParallelComponent::getPort(const std::string& name)
{
    map<string, PortRecord*>::iterator iter = portmap.find(name);
    if(iter == portmap.end())
	return 0;
    else if(iter->second->connections.size()> 1)
	return 0;
    else
	return iter->second->connections[0];
}

void UintahParallelComponent::releasePort(const std::string&)
{
}


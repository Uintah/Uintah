/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <SCICore/Util/NotFinished.h>
#include <algorithm>
using std::map;
using std::find;

using Uintah::Parallel::UintahParallelComponent;
using Uintah::Parallel::UintahParallelPort;

UintahParallelComponent::UintahParallelComponent()
{
}

UintahParallelComponent::~UintahParallelComponent()
{
}

void
UintahParallelComponent::attachPort(const string& name,
				    UintahParallelPort* port)
{
    map<string, PortRecord*>::iterator iter = portmap.find(name);
    if(iter == portmap.end()){
	portmap[name]=new PortRecord(port);
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

//
// $Log$
// Revision 1.4  2000/04/12 23:01:03  sparker
// Make getPort work
//
// Revision 1.3  2000/04/11 07:10:56  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.2  2000/03/16 22:08:39  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

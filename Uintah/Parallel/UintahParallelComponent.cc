/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>
#include <algorithm>

using namespace Uintah;
using namespace std;

UintahParallelComponent::UintahParallelComponent(const ProcessorGroup* myworld)
   : d_myworld(myworld)
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

//
// $Log$
// Revision 1.8  2000/06/17 07:06:49  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.7  2000/05/30 20:19:43  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.6  2000/04/26 06:49:16  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/19 21:20:05  dav
// more MPI stuff
//
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

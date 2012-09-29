/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Malloc/Allocator.h>
#include <algorithm>

using namespace Uintah;
using std::map;
using std::string;

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
	return iter->second->connections.back();
    else
	return iter->second->connections[0];
}

UintahParallelPort* UintahParallelComponent::getPort(const std::string& name,
                                                     unsigned int i)
{
    map<string, PortRecord*>::iterator iter = portmap.find(name);
    if(iter == portmap.end())
	return 0;
    else if(iter->second->connections.size()> 1)
	return iter->second->connections[i];
    else
	return iter->second->connections[0];
}

void UintahParallelComponent::releasePort(const std::string&)
{
}

unsigned int UintahParallelComponent::numConnections(const std::string& name)
{
  map<string, PortRecord*>::iterator iter = portmap.find(name);
  if(iter == portmap.end())
    return 0;
  else 
    return iter->second->connections.size();
}

/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  CCAPortInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/CCA/CCAPortInstance.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

CCAPortInstance::CCAPortInstance(const std::string& name,
				 const std::string& type,
				 const sci::cca::TypeMap::pointer& properties,
				 PortType porttype)
  : name(name), type(type), properties(properties), porttype(porttype),
    useCount(0)
{
}

CCAPortInstance::CCAPortInstance(const std::string& name,
				 const std::string& type,
				 const sci::cca::TypeMap::pointer& properties,
				 const sci::cca::Port::pointer& port,
				 PortType porttype)
  : name(name), type(type), properties(properties), port(port),
    porttype(porttype), useCount(0)
{
}

CCAPortInstance::~CCAPortInstance()
{
}

bool CCAPortInstance::connect(PortInstance* to)
{
  if(!canConnectTo(to))
    return false;
  //CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  PortInstance* p2 = to;
  if(!p2)
    return false;
  if(portType() == From && p2->portType() == To){
    connections.push_back(p2);
  } 
  else p2->connect(this);
  return true;
}

PortInstance::PortType CCAPortInstance::portType()
{
  if(porttype == Uses)
    return From;
  else
    return To;
}


std::string CCAPortInstance::getType()
{
  return type;
}

std::string CCAPortInstance::getModel()
{
  return "cca"; 
}


string CCAPortInstance::getUniqueName()
{
  // CCA names are already guaranteed to be unique
  return name;
}

bool CCAPortInstance::disconnect(PortInstance* to)
{
  CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  if(!p2)
    return false;

  if(porttype !=Uses){
    cerr<<"disconnect can be called only by user"<<endl; 
    return false;
  } 
  std::vector<PortInstance*>::iterator iter;
  for(iter=connections.begin(); iter<connections.end();iter++){
    if(p2==(*iter)){
      connections.erase(iter);
      return true;
    }
  }
  return false;
}

bool CCAPortInstance::canConnectTo(PortInstance* to)
{
  //CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  PortInstance* p2 = to;
  if( p2 && getType()==p2->getType() && portType()!=p2->portType() ){
    if(available() && p2->available()) return true;
  }
  return false;
}

bool CCAPortInstance::available()
{
  return portType()==To || connections.size()==0;
}

PortInstance* CCAPortInstance::getPeer()
{
  return connections[0];
}

string CCAPortInstance::getName()
{
  return name;
}

void CCAPortInstance::incrementUseCount()
{
  useCount++;
}

bool CCAPortInstance::decrementUseCount()
{
  if(useCount<=0)
    return false;
  useCount--;
  return true;
}

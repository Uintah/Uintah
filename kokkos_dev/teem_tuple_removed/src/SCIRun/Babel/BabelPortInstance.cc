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
 *  BabelPortInstance.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 */

#include <SCIRun/Babel/BabelPortInstance.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

BabelPortInstance::BabelPortInstance(const std::string& name,
				 const std::string& type,
				 const gov::cca::TypeMap& properties,
				 PortType porttype)
  : porttype(porttype), name(name), type(type), properties(properties),
    useCount(0)
{
}

BabelPortInstance::BabelPortInstance(const std::string& name,
				 const std::string& type,
				 const gov::cca::TypeMap& properties,
				 const gov::cca::Port& port,
				 PortType porttype)
  : porttype(porttype), name(name), type(type), properties(properties),
    port(port), useCount(0)
{
}

BabelPortInstance::~BabelPortInstance()
{
}

bool BabelPortInstance::connect(PortInstance* to)
{
  if(!canConnectTo(to))
    return false;
  //BabelPortInstance* p2 = dynamic_cast<BabelPortInstance*>(to);
  PortInstance* p2 = to;
  if(!p2)
    return false;
  if(portType() == From && p2->portType() == To){
    connections.push_back(p2);
  }
  else p2->connect(this);
  return true;
}

PortInstance::PortType BabelPortInstance::portType()
{
  if(porttype == Uses)
    return From;
  else
    return To;
}

std::string BabelPortInstance::getType()
{
  return type;
}

std::string BabelPortInstance::getModel()
{
  return "babel"; 
}

string BabelPortInstance::getUniqueName()
{
  // Babel names are already guaranteed to be unique
  return name;
}

bool BabelPortInstance::disconnect(PortInstance* to)
{
  BabelPortInstance* p2 = dynamic_cast<BabelPortInstance*>(to);
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

bool BabelPortInstance::canConnectTo(PortInstance* to)
{
  //BabelPortInstance* p2 = dynamic_cast<BabelPortInstance*>(to);
  PortInstance* p2 = to;
  //  cerr<<"try to connect:"<<endl;
  //cerr<<"type(p1)="<<type<<endl;
  //cerr<<"type(p2)="<<p2->type<<endl;
  //cerr<<"port type(p1)="<<porttype<<endl;
  //cerr<<"port type(p2)="<<p2->porttype<<endl;
  
  if( p2 && getType()==p2->getType() && portType()!=p2->portType() ){
    if(available() && p2->available()) return true;
  }
  return false;
}

bool BabelPortInstance::available()
{
  return portType()==To || connections.size()==0;
}

PortInstance* BabelPortInstance::getPeer()
{
  return connections[0];
}

string BabelPortInstance::getName()
{
  return name;
}

void BabelPortInstance::incrementUseCount()
{
  useCount++;
}

bool BabelPortInstance::decrementUseCount()
{
  if(useCount<=0)
    return false;
  useCount--;
  return true;
}

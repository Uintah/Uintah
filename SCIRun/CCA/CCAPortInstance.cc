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
				 const gov::cca::TypeMap::pointer& properties,
				 PortType porttype)
  : name(name), type(type), properties(properties), porttype(porttype)
{
}

CCAPortInstance::CCAPortInstance(const std::string& name,
				 const std::string& type,
				 const gov::cca::TypeMap::pointer& properties,
				 const gov::cca::Port::pointer& port,
				 PortType porttype)
  : name(name), type(type), properties(properties), port(port),
    porttype(porttype)
{
}

CCAPortInstance::~CCAPortInstance()
{
}

bool CCAPortInstance::connect(PortInstance* to)
{
  CCAPortInstance* p2 = dynamic_cast<CCAPortInstance*>(to);
  if(!p2)
    return false;
  cerr << "Must have better port type checking on connect!\n";
  if(type != p2->type){
    cerr << "connect type mismatch: " << type << " and " << p2->type << '\n';
    return false;
  }
  if(porttype == Uses && p2->porttype == Provides){
    connections.push_back(p2);
    return true;
  } else if(porttype == Provides && p2->porttype == Uses){
    p2->connections.push_back(this);
    return true;
  }
  return false;
}

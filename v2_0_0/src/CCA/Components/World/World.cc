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
 *  World.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#include <CCA/Components/World/World.h>
#include <iostream>

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_World()
{
  return sci::cca::Component::pointer(new World());
}


World::World(){
}

World::~World(){
}

void World::setServices(const sci::cca::Services::pointer& svc){

  services=svc;
  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  StringPort::pointer strport=StringPort::pointer(new StringPort);
  svc->addProvidesPort(strport,"stringport","sci.cca.ports.StringPort", props);
}


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
 *  Hello.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#include <CCA/Components/Hello/Hello.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Hello()
{
  return sci::cca::Component::pointer(new Hello());
}


Hello::Hello(){
}

Hello::~Hello(){
}

void Hello::setServices(const sci::cca::Services::pointer& svc){
  services=svc;
  cerr<<"svc->createTypeMap...";
  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  cerr<<"Done\n";

  myUIPort::pointer uip=myUIPort::pointer(new myUIPort);
  myGoPort::pointer gop=myGoPort::pointer(new myGoPort(svc));

  cerr<<"svc->addProvidesPort(uip)...";  
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
  cerr<<"Done\n";

  cerr<<"svc->addProvidesPort(gop)...";  
  svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort", props);
  cerr<<"Done\n";

  svc->registerUsesPort("stringport","sci.cca.ports.StringPort", props);
}

int myUIPort::ui(){
  cout<<"UI button is clicked!\n";
  return 0;
}

myGoPort::myGoPort(const sci::cca::Services::pointer& svc){
  this->services=svc;
}

int myGoPort::go(){
  if(services.isNull()){
    cerr<<"Null services!\n";
    return 1;
  }
  cerr<<"Hello.go.getPort...";
  sci::cca::Port::pointer pp=services->getPort("stringport");	
  cerr<<"Done\n";
  if(pp.isNull()){
    cerr<<"stringport is not available!\n";
    return 1;
  }  
  else{
    cerr<<"stringport is not null\n";
  }
  cerr<<"Hello.go.pidl_cast...";
  sci::cca::ports::StringPort::pointer sp=pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
  cerr<<"Done\n";

  cerr<<"Hello.go.port.getString...";
  std::string name=sp->getString();
  cerr<<"Done\n";


  services->releasePort("stringport");

  cout<<"Hello "<<name<<endl;
  return 0;
}
 

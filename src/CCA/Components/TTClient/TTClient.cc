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
 *  TTClient.cc:
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#include <CCA/Components/TTClient/TTClient.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

//#include <qapplication.h>
//#include <qpushbutton.h>
//#include <qmessagebox.h>



using namespace std;
using namespace SCIRun;

extern "C" gov::cca::Component::pointer make_SCIRun_TTClient()
{
  return gov::cca::Component::pointer(new TTClient());
}


TTClient::TTClient()
{
  uiPort.setParent(this);
  goPort.setParent(this);
}

TTClient::~TTClient()
{

}

void TTClient::setServices(const gov::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  ttUIPort::pointer uip(&uiPort);
  ttGoPort::pointer gop(&goPort);
  svc->addProvidesPort(uip,"ui","gov.cca.UIPort", props);
  svc->addProvidesPort(gop,"go","gov.cca.GoPort", props);
  svc->registerUsesPort("tt","gov.cca.TTPort", props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

int ttUIPort::ui() 
{
//  QMessageBox::warning(0, "TTClient", "You have clicked the UI button!");
  return 0;
}


int ttGoPort::go() 
{
  //QMessageBox::warning(0, "TTClient", "Go ...");
  cout<<"GoGoGo!"<<endl;
  gov::cca::Port::pointer pp=TTCl->getServices()->getPort("tt");
  if(pp.isNull()){
    //QMessageBox::warning(0, "Tri", "Port tt is not available!");
    cout<<"pp_isNULL"<<endl;
    return 1;
  }

  PP::PingPong::pointer PPptr=
       pidl_cast<PP::PingPong::pointer>(pp);

  if(PPptr.isNull()) {
    cout<<"PPptr_isNULL"<<endl;
  } 

  PPptr->pingpong(13);
  cout<<"PPptr-All_WELL_AND_DONE"<<endl;

  return 0;
}
 

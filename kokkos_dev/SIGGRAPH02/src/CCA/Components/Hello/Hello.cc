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

#include <qapplication.h>
#include <qpushbutton.h>
#include <qmessagebox.h>



using namespace std;
using namespace SCIRun;

extern "C" gov::cca::Component::pointer make_SCIRun_Hello()
{
  return gov::cca::Component::pointer(new Hello());
}


Hello::Hello()
{

}

Hello::~Hello()
{
  cerr << "called ~Hello()\n";
}

void Hello::setServices(const gov::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  myGoPort::pointer gop(&goPort);
  svc->addProvidesPort(uip,"ui","gov.cca.UIPort", props);
  svc->addProvidesPort(gop,"go","gov.cca.GoPort", props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

int myUIPort::ui() 
{
  QMessageBox::warning(0, "Hello", "You have clicked the UI button!");
  return 0;
}


int myGoPort::go() 
{
  QMessageBox::warning(0, "Hello", "Go ...");
  return 0;
}
 

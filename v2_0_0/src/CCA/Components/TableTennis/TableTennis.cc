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
 *  TableTennis.cc:
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#include <CCA/Components/TableTennis/TableTennis.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

//#include <qapplication.h>
//#include <qpushbutton.h>
//#include <qmessagebox.h>



using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TableTennis()
{
  return sci::cca::Component::pointer(new TableTennis());
}


TableTennis::TableTennis()
{

}

TableTennis::~TableTennis()
{
  cerr << "called ~TableTennis()\n";
}

void TableTennis::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  myGoPort::pointer gop(&goPort);
  myTTPort::pointer ttp(&ttPort);
  svc->addProvidesPort(uip,"ui","sci.cca.UIPort", props);
  svc->addProvidesPort(gop,"go","sci.cca.GoPort", props);
  svc->addProvidesPort(ttp,"tt","sci.cca.TTPort", props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

int myUIPort::ui() 
{
//  QMessageBox::warning(0, "TableTennis", "You have clicked the UI button!");
  return 0;
}


int myGoPort::go() 
{
  //QMessageBox::warning(0, "TableTennis", "Go ...");
  cout<<"Nowhere to go!"<<endl;
  return 0;
}

int myTTPort::pingpong(int test)
{
  cout << "Test = " << test << "\n";
  return test;
}


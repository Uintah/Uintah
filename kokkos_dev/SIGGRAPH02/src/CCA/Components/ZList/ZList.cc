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
 *  ZList.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */

#include <iostream.h>
#include <qapplication.h>
#include <qpushbutton.h>
#include <qmessagebox.h>
#include "ZList.h"
//#include "ListForm.h"
extern "C" gov::cca::Component::pointer make_SCIRun_ZList()
{
  return gov::cca::Component::pointer(new ZList());
}


ZList::ZList()
{
	uiport.setParent(this);
	listport.setParent(this);
}

ZList::~ZList()
{
}

void ZList::setServices(const gov::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  ImUIPort1::pointer uip(&uiport);
  ImZListPort::pointer zlp(&listport);
  svc->addProvidesPort(uip,"ui","gov.cca.UIPort", props);
  svc->addProvidesPort(zlp,"listport","ZListPort", props);
}

int ImUIPort1::ui()
{
  ListForm *w = new ListForm(com);
  w->show();
  return 0;
}

CIA::array1<double> ImZListPort::getList()
{
 CIA::array1<double> data;
 return com->datalist;
}

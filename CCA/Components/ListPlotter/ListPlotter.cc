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
 *  ListPlotter.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */

#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>
#include <qapplication.h>
#include <qpushbutton.h>
#include <qmessagebox.h>
#include "ListPlotter.h"
#include "ListPlotterForm.h"

extern "C" gov::cca::Component::pointer make_SCIRun_ListPlotter()
{
  return gov::cca::Component::pointer(new ListPlotter());
}


ListPlotter::ListPlotter()
{

}

ListPlotter::~ListPlotter()
{
}

void ListPlotter::setServices(const gov::cca::Services::pointer& svc)
{
  //cerr<<"ListPlotter::serService is  called#################\n";

  services=svc;
  //register provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  ImUIPort::pointer uip(&ui);
	ImUIPort::pointer gop(&ui);
  svc->addProvidesPort(uip,"ui","UIPort", props);
  svc->addProvidesPort(gop,"go","GoPort", props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
}

void ImUIPort::ui()
{
  ListPlotterForm *w = new ListPlotterForm;
		
  const int size=8;
  double val[size]={1,2,5,3,7,4,3,2};
  w->setData(val, size);
  w->show();
  //delete w;	
}

int ImGoPort::go()
{
  QMessageBox::warning(0, "ImGoPort", "go!");
  return 0;	
}

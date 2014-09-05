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

extern "C" sci::cca::Component::pointer make_SCIRun_ListPlotter()
{
  return sci::cca::Component::pointer(new ListPlotter());
}


ListPlotter::ListPlotter()
{
	
}

ListPlotter::~ListPlotter()
{
}

void ListPlotter::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  ui.setServices(svc);	
  //register provides ports here ...  

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  ImUIPort::pointer uip(&ui);
	ImUIPort::pointer gop(&ui);
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
  svc->registerUsesPort("listport","ZListPort", props);
}

void ImUIPort::setServices(const sci::cca::Services::pointer& svc)
{
	services=svc;
}

int ImUIPort::ui()
{
  
  ListPlotterForm *w = new ListPlotterForm; 
  sci::cca::Port::pointer pp=services->getPort("listport");	
  if(pp.isNull()){
    QMessageBox::warning(0, "ListPlotter", "listport is not available!");
    return 1;
  }  
  sci::cca::ports::ZListPort::pointer lport=pidl_cast<sci::cca::ports::ZListPort::pointer>(pp);
  SSIDL::array1<double> data=lport->getList();	

  services->releasePort("listport");

  int size=data.size();
  double *val=new double[size]; 	
  for(int i=0; i<size; i++){
	val[i]=data[i];	
  }
  w->setData(val, size);
  w->show();
  delete val;
  return 0;
}


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
 *  Viewer.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/Viewer/Viewer.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

#include <qapplication.h>
#include <qpushbutton.h>
#include <qmainwindow.h>
#include <qmessagebox.h>
#include <qapplication.h>
#include <qframe.h>
#include <qpainter.h>
#include <qcolor.h>
#include <qdialog.h>

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "vector2d.h"
#include "MainWindow.h"
using namespace std;
using namespace SCIRun;

extern "C" gov::cca::Component::pointer make_SCIRun_Viewer()
{
  return gov::cca::Component::pointer(new Viewer());
}

Viewer::Viewer()
{
  uiPort.setParent(this);
}

Viewer::~Viewer()
{
  cerr << "called ~Viewer()\n";
}

void Viewer::setServices(const gov::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  gov::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  svc->addProvidesPort(uip,"ui","gov.cca.UIPort", props);
  svc->registerUsesPort("field", "gov.cca.Field2DPort",props);
  svc->registerUsesPort("mesh", "gov.cca.MeshPort",props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

int myUIPort::ui() 
{
  gov::cca::Port::pointer pp=com->getServices()->getPort("mesh");	
  if(pp.isNull()){
    QMessageBox::warning(0, "Viewer", "Port mesh is not available!");
    return 1;
  }  
  gov::cca::ports::MeshPort::pointer pdePort=
    pidl_cast<gov::cca::ports::MeshPort::pointer>(pp);
  SIDL::array1<double> nodes=pdePort->getNodes();	
  SIDL::array1<int> triangles=pdePort->getTriangles();	
  com->getServices()->releasePort("mesh");	

  gov::cca::Port::pointer pp2=com->getServices()->getPort("field");	
  if(pp2.isNull()){
    QMessageBox::warning(0, "Viewer", "field is not available!");
    return 1;
  }  
  gov::cca::ports::Field2DPort::pointer fport=
    pidl_cast<gov::cca::ports::Field2DPort::pointer>(pp2);
  SIDL::array1<double> solution=fport->getField();	
  com->getServices()->releasePort("field");	


  if(nodes.size()/2 !=solution.size()){
    QMessageBox::warning(0,"Viewer","Mesh and Field do not match!");
    return 1;
  }

  (new MainWindow(0, 0, nodes,triangles,solution))->show();
  return 0;
}


 





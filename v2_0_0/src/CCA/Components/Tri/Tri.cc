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
 *  Tri.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/Tri/Tri.h>
#include <iostream>
#include <CCA/Components/Builder/QtUtils.h>

#include <qapplication.h>
#include <qpushbutton.h>
#include <qmessagebox.h>
#include "MeshWindow.h"


using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Tri()
{
  return sci::cca::Component::pointer(new Tri());
}


Tri::Tri()
{
  uiPort.setParent(this);
  goPort.setParent(this);
  meshPort.setParent(this);
  mesh=0;
}

Tri::~Tri()
{
  cerr << "called ~Tri()\n";
  if(mesh!=0) delete mesh;
}

void Tri::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  myGoPort::pointer gop(&goPort);
  myMeshPort::pointer meshp(&meshPort);
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
  svc->addProvidesPort(gop,"go","sci.cca.ports.GoPort", props);
  svc->addProvidesPort(meshp,"mesh","sci.cca.ports.MeshPort", props);
  svc->registerUsesPort("pde","sci.cca.ports.PDEDescriptionPort", props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

int myUIPort::ui() 
{
  if(com->mesh==0) return 1;
  (new MeshWindow(0, com->mesh ))->show();
  return 0;
}


int myGoPort::go() 
{
  sci::cca::Port::pointer pp=com->getServices()->getPort("pde");	
  if(pp.isNull()){
    QMessageBox::warning(0, "Tri", "Port pde is not available!");
    return 1;
  }  
  sci::cca::ports::PDEDescriptionPort::pointer pdePort=
    pidl_cast<sci::cca::ports::PDEDescriptionPort::pointer>(pp);
  SSIDL::array1<double> nodes1d=pdePort->getNodes();
  SSIDL::array1<int> boundaries1d=pdePort->getBoundaries();

  com->getServices()->releasePort("pde");

  if(com->mesh!=0) delete com->mesh;

  com->mesh=new Delaunay(nodes1d, boundaries1d);

  com->mesh->triangulation();

  return 0;
}

SSIDL::array1<int> myMeshPort::getTriangles()
{
  SSIDL::array1<int> vindex;
  if(com->mesh!=0){
    std::vector<Triangle> tri=com->mesh->getTriangles();

    for(unsigned int i=0; i<tri.size();i++){
      vindex.push_back(tri[i].index[0]-4);
      vindex.push_back(tri[i].index[1]-4);
      vindex.push_back(tri[i].index[2]-4);
    }
  }
  return vindex;
}

SSIDL::array1<double> myMeshPort::getNodes()
{
  sci::cca::Port::pointer pp=com->getServices()->getPort("pde");	
  if(pp.isNull()){
    QMessageBox::warning(0, "Tri", "Port pde is not available!");
    return 1;
  }  
  sci::cca::ports::PDEDescriptionPort::pointer pdePort=
    pidl_cast<sci::cca::ports::PDEDescriptionPort::pointer>(pp);
  SSIDL::array1<double> nodes1d=pdePort->getNodes();

  com->getServices()->releasePort("pde");	
  
  return nodes1d;
}
 



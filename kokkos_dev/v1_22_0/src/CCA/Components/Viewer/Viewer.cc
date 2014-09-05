/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

extern "C" sci::cca::Component::pointer make_SCIRun_Viewer()
{
  return sci::cca::Component::pointer(new Viewer());
}

Viewer::Viewer()
{
  uiPort.setParent(this);
}

Viewer::~Viewer()
{
  cerr << "called ~Viewer()\n";
}

void Viewer::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
  svc->registerUsesPort("field", "sci.cca.ports.Field2DPort",props);
  svc->registerUsesPort("mesh", "sci.cca.ports.MeshPort",props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

int myUIPort::ui() 
{
  sci::cca::Port::pointer pp=com->getServices()->getPort("mesh");	
  if(pp.isNull()){
    QMessageBox::warning(0, "Viewer", "Port mesh is not available!");
    return 1;
  }  
  sci::cca::ports::MeshPort::pointer pdePort=
    pidl_cast<sci::cca::ports::MeshPort::pointer>(pp);
  SSIDL::array1<double> nodes=pdePort->getNodes();	
  SSIDL::array1<int> triangles=pdePort->getTriangles();	
  com->getServices()->releasePort("mesh");	

  sci::cca::Port::pointer pp2=com->getServices()->getPort("field");	
  if(pp2.isNull()){
    QMessageBox::warning(0, "Viewer", "field is not available!");
    return 1;
  }  
  sci::cca::ports::Field2DPort::pointer fport=
    pidl_cast<sci::cca::ports::Field2DPort::pointer>(pp2);
  SSIDL::array1<double> solution=fport->getField();	
  com->getServices()->releasePort("field");	


  if(nodes.size()/2 !=solution.size()){
    QMessageBox::warning(0,"Viewer","Mesh and Field do not match!");
    return 1;
  }

  (new MainWindow(0, 0, nodes,triangles,solution))->show();
  return 0;
}


 





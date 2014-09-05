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

Viewer::Viewer(){
}

Viewer::~Viewer(){
}

void Viewer::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  myViewPort::pointer uip(new myViewPort);
  svc->addProvidesPort(uip,"viewer","sci.cca.ports.ViewPort",   sci::cca::TypeMap::pointer(NULL));
}

int 
myViewPort::view2dPDE(const SSIDL::array1<double> &nodes, 
		      const SSIDL::array1<int> &triangles, 
		      const SSIDL::array1<double> &solution){

  if(nodes.size()/2 !=solution.size()){
    QMessageBox::warning(0,"Viewer","Mesh and Field do not match!");
    return 1;
  }

  (new MainWindow(0, 0, nodes,triangles,solution))->show();
  return 0;
}


 





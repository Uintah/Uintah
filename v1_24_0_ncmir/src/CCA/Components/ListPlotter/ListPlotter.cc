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


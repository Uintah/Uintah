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
 *  ZList.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */

#include <CCA/Components/ZList/ZList.h>

#include <iostream>
#include <qapplication.h>
#include <qpushbutton.h>
#include <qmessagebox.h>

//#include "ListForm.h"

extern "C" sci::cca::Component::pointer make_SCIRun_ZList()
{
  return sci::cca::Component::pointer(new ZList());
}


ZList::ZList()
{
  uiport.setParent(this);
  listport.setParent(this);
}

ZList::~ZList()
{
}

void ZList::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  ImUIPort1::pointer uip(&uiport);
  ImZListPort::pointer zlp(&listport);
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
  svc->addProvidesPort(zlp,"listport","ZListPort", props);
}

int ImUIPort1::ui()
{
  ListForm *w = new ListForm(com);
  w->show();
  return 0;
}

SSIDL::array1<double> ImZListPort::getList()
{
  SSIDL::array1<double> data;
  return com->datalist;
}

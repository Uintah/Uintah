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


#include <CCA/Components/ListPlotter/ListPlotter.h>
#include <CCA/Components/ListPlotter/ListPlotterForm.h>

#include <iostream>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_ListPlotter()
{
  return sci::cca::Component::pointer(new ListPlotter());
}


ListPlotter::ListPlotter()
{
}

ListPlotter::~ListPlotter()
{
  services->unregisterUsesPort("listport");
  services->removeProvidesPort("ui");
}

void ListPlotter::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  ImUIPort::pointer uip = ImUIPort::pointer(new ImUIPort(svc));

  svc->addProvidesPort(uip, "ui", "sci.cca.ports.UIPort", props);
  svc->registerUsesPort("listport", "ZListPort", props);
}

int ImUIPort::ui()
{

  ListPlotterForm *w = new ListPlotterForm;
  sci::cca::ports::ZListPort::pointer lport;
  try {
    sci::cca::Port::pointer pp = services->getPort("listport");
    lport = pidl_cast<sci::cca::ports::ZListPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(STLTowxString(e->getNote()), wxT("ListPlotter"), wxOK|wxICON_ERROR, 0);
    return -1;
  }
  SSIDL::array1<double> data = lport->getList();
  services->releasePort("listport");

  int size = data.size();
  double *val = new double[size];
  for (int i = 0; i < size; i++) {
    val[i] = data[i];
  }
  w->setData(val, size);
  // could use ShowModal...
  w->Show();
  delete [] val;
  return 0;
}

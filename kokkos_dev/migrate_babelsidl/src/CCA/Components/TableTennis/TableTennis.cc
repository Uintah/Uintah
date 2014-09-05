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
 *  TableTennis.cc:
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#include <CCA/Components/TableTennis/TableTennis.h>
#include <iostream>
#include <sci_wx.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TableTennis()
{
  return sci::cca::Component::pointer(new TableTennis());
}


TableTennis::TableTennis()
{
}

TableTennis::~TableTennis()
{
  services->removeProvidesPort("tt");
}

void TableTennis::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  //register provides ports here ...

  sci::cca::TypeMap::pointer props = services->createTypeMap();
  myTTPort::pointer ttp(new myTTPort);
  services->addProvidesPort(ttp, "tt", "sci.cca.ports.TTPort", props);
}

int myTTPort::pingpong(int test)
{
#if HAVE_GUI
  wxString msg(wxT("Test = "));
  msg += wxString::Format(wxT("%d"), test);
  wxMessageBox(msg, wxT("TableTennis"), wxOK|wxICON_INFORMATION, 0);
#else
  std::cout << "Test = " << test << std::endl;
#endif
  return test;
}

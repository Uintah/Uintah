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
#include <CCA/Components/Builder/QtUtils.h>
#include <iostream>


using namespace std;
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

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
<<<<<<< .working
  myUIPort::pointer uip(&uiPort);
  myGoPort::pointer gop(&goPort);
  myTTPort::pointer ttp(&ttPort);
  svc->addProvidesPort(uip,"ui","sci.cca.UIPort", props);
  svc->addProvidesPort(gop,"go","sci.cca.GoPort", props);
  svc->addProvidesPort(ttp,"tt","sci.cca.TTPort", props);
=======
  myTTPort::pointer ttp(new myTTPort);
  svc->addProvidesPort(ttp,"tt","sci.cca.ports.TTPort", props);
>>>>>>> .merge-right.r32054
}

int myTTPort::pingpong(int test)
{
  cout << "Test = " << test << "\n";
  return test;
}


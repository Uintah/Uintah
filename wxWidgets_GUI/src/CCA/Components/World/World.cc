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
 *  World.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#include <sci_wx.h>
#include <CCA/Components/World/World.h>
#include <iostream>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_World()
{
    return sci::cca::Component::pointer(new World());
}


World::World(): text("World")
{
}

World::~World()
{
    services->removeProvidesPort("stringport");
    services->removeProvidesPort("ui");
    services->removeProvidesPort("icon");
}

void World::setServices(const sci::cca::Services::pointer& svc)
{
    services = svc;
    ComponentIcon* ci = new ComponentIcon();
    StringPort *sp = new StringPort();
    sp->setParent(this);
    svc->addProvidesPort(StringPort::pointer(sp), "stringport",
                         "sci.cca.ports.StringPort", svc->createTypeMap());

    WUIPort *uip = new WUIPort(ci->getDisplayName());
    uip->setParent(this);
    svc->addProvidesPort(WUIPort::pointer(uip),"ui","sci.cca.ports.UIPort", svc->createTypeMap());

    svc->addProvidesPort(ComponentIcon::pointer(ci), "icon",
                         "sci.cca.ports.ComponentIcon", svc->createTypeMap());
}

int WUIPort::ui()
{
#if HAVE_WX
  wxString t = wxGetTextFromUser(wxT("Enter some text"), displayName);
  com->setText(t.c_str());
#endif
    return 0;
}

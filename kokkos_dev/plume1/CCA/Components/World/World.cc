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

#include <sci_defs/qt_defs.h>
#include <CCA/Components/World/World.h>
#include <iostream>

#if HAVE_QT
 #include <qinputdialog.h>
 #include <qstring.h>
#endif

//using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_World()
{
    return sci::cca::Component::pointer(new World());
}


World::World()
{
    strPort.setParent(this);
    uiPort.setParent(this);
    ciPort.setParent(this);
    text = "World";
}

World::~World()
{
}

void World::setServices(const sci::cca::Services::pointer& svc)
{
    services = svc;
    sci::cca::TypeMap::pointer props = svc->createTypeMap();
    StringPort::pointer strp(&strPort);
    WUIPort::pointer uip(&uiPort);
    ComponentIcon::pointer cip(&ciPort);

    svc->addProvidesPort(strp, "stringport",
                         "sci.cca.ports.StringPort", props);
    svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
    svc->addProvidesPort(cip, "icon",
                         "sci.cca.ports.ComponentIcon", props);
    //svc->registerUsesPort();
}

std::string StringPort::getString() { return com->text; }

int WUIPort::ui()
{
#if HAVE_QT
    bool ok;
    QString t = QInputDialog::getText("World", "Enter some text:",
        QLineEdit::Normal, QString::null, &ok);
    if (ok && !t.isEmpty()) {
        com->text = t.ascii();
    }
#endif
    return 0;
}


std::string ComponentIcon::getDisplayName()
{
    return "World";
}

std::string ComponentIcon::getDescription()
{
    return "The World component is a sample CCA component that provides a sci::cca::StringPort.";
}

int ComponentIcon::getProgressBar()
{
    return 0;
}
 
std::string ComponentIcon::getIconShape()
{
    return "RECT";
}
 

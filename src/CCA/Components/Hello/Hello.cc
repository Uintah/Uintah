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
 *  Hello.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#include <sci_defs/qt_defs.h>
#include <CCA/Components/Hello/Hello.h>
#include <CCA/Components/Builder/QtUtils.h>
#include <Core/Thread/Time.h>
#include <SCIRun/TypeMap.h>

#include <iostream>

#if HAVE_QT
 #include <qmessagebox.h>
 #include <qstring.h>
#endif

#include <unistd.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Hello()
{
    return sci::cca::Component::pointer(new Hello());
}


Hello::Hello() : text("GO hasn't been called yet!")
{
}

Hello::~Hello()
{
}

void Hello::setServices(const sci::cca::Services::pointer& svc)
{
    services = svc;
    svc->registerForRelease(sci::cca::ComponentRelease::pointer(this));
    sci::cca::TypeMap::pointer props = svc->createTypeMap();

    myUIPort *uip = new myUIPort();
    uip->setParent(this);
    myUIPort::pointer uiPortPtr = myUIPort::pointer(uip);

    svc->addProvidesPort(uiPortPtr, "ui", "sci.cca.ports.UIPort",
                         sci::cca::TypeMap::pointer(0));

    myGoPort *gp = new myGoPort(svc);
    gp->setParent(this);
    myGoPort::pointer goPortPtr = myGoPort::pointer(gp);

    svc->addProvidesPort(goPortPtr, "go", "sci.cca.ports.GoPort",
                         sci::cca::TypeMap::pointer(0));

    myComponentIcon::pointer ciPortPtr = myComponentIcon::pointer(new myComponentIcon);

    svc->addProvidesPort(ciPortPtr, "icon", "sci.cca.ports.ComponentIcon",
                         sci::cca::TypeMap::pointer(0));

    props->putString("cca.portName", "stringport");
    props->putString("cca.portType", "sci.cca.ports.StringPort");
    svc->registerUsesPort("stringport","sci.cca.ports.StringPort", props);

    sci::cca::TypeMap::pointer props2 = svc->createTypeMap();
    svc->registerUsesPort("progress","sci.cca.ports.Progress", props2);
}

void Hello::releaseServices(const sci::cca::Services::pointer& svc)
{
std::cerr << "Hello::releaseServices" << std::endl;

    svc->unregisterUsesPort("stringport");
    svc->unregisterUsesPort("progress");

    svc->removeProvidesPort("ui");
    svc->removeProvidesPort("go");
    svc->removeProvidesPort("icon");
}


int myUIPort::ui()
{
#if HAVE_QT
    QMessageBox::information(0, "Hello", com->text);
#else
    std::cerr << "UI not available." << std::endl;
#endif
    return 0;
}

myGoPort::myGoPort(const sci::cca::Services::pointer& svc)
{
    this->services = svc;
}

int myGoPort::go()
{
    if (services.isNull()) {
        std::cerr << "Null services!\n";
        return 1;
    }
    std::cerr << "Hello.go.getPort...";
    double st = SCIRun::Time::currentSeconds();

    sci::cca::Port::pointer pp;
    sci::cca::Port::pointer progPort;
    try {
        pp = services->getPort("stringport");
        progPort = services->getPort("progress");
    }
    catch (const sci::cca::CCAException::pointer &e) {
        std::cerr << e->getNote() << std::endl;
        return 1;
    }

    sci::cca::ports::Progress::pointer pPtr =
        pidl_cast<sci::cca::ports::Progress::pointer>(progPort);

    sci::cca::ports::StringPort::pointer sp =
        pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
    std::string name = sp->getString();

    double t = Time::currentSeconds() - st;
    std::cerr << "Done in " << t << "secs\n";
    std::cerr << t*1000*1000 << " us/rep\n";

    if (! name.empty()) {
      com->text = name.c_str();
    }

    pPtr->updateProgress(myComponentIcon::STEPS);
    services->releasePort("stringport");
    services->releasePort("progress");

    return 0;
}

std::string myComponentIcon::getDisplayName()
{
    return std::string("Hello Component");
}

std::string myComponentIcon::getDescription()
{
    return std::string("The Hello component is a sample CCA component that uses a sci::cca::StringPort.");
}

int myComponentIcon::getProgressBar()
{
    return STEPS;
}

std::string myComponentIcon::getIconShape()
{
    return std::string("RECT");
}


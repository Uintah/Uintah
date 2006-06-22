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

#include <sci_wx.h>
#include <CCA/Components/Hello/Hello.h>
#include <Core/Thread/Time.h>
#include <SCIRun/TypeMap.h>

#include <iostream>
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

    HelloUIPort *uip = new HelloUIPort(svc);
    uip->setParent(this);
    HelloUIPort::pointer uiPortPtr = HelloUIPort::pointer(uip);

    svc->addProvidesPort(uiPortPtr, "ui", "sci.cca.ports.UIPort",
                         sci::cca::TypeMap::pointer(0));

    HelloGoPort *gp = new HelloGoPort(svc);
    gp->setParent(this);
    HelloGoPort::pointer goPortPtr = HelloGoPort::pointer(gp);

    svc->addProvidesPort(goPortPtr, "go", "sci.cca.ports.GoPort",
                         sci::cca::TypeMap::pointer(0));

    props->putString("cca.portName", "stringport");
    props->putString("cca.portType", "sci.cca.ports.StringPort");
    svc->registerUsesPort("stringport","sci.cca.ports.StringPort", props);

//     HelloComponentIcon::pointer ciPortPtr = HelloComponentIcon::pointer(new HelloComponentIcon);
//     svc->addProvidesPort(ciPortPtr, "icon", "sci.cca.ports.ComponentIcon",
//                          sci::cca::TypeMap::pointer(0));

//     sci::cca::TypeMap::pointer props2 = svc->createTypeMap();
//     svc->registerUsesPort("progress","sci.cca.ports.Progress", props2);
}

void Hello::releaseServices(const sci::cca::Services::pointer& svc)
{
std::cerr << "Hello::releaseServices" << std::endl;

    services->unregisterUsesPort("stringport");
    //services->unregisterUsesPort("progress");

    //services->removeProvidesPort("ui");
    services->removeProvidesPort("go");
    //services->removeProvidesPort("icon");
}


int HelloUIPort::ui()
{
// #if HAVE_QT
//     QMessageBox::information(0, "Hello", com->text);
// #else
//    std::cerr << "UI not available." << std::endl;
// #endif
#if HAVE_WX
//   try {
//     sci::cca::ports::GUIService::pointer guiService =
//       pidl_cast<sci::cca::ports::GUIService::pointer>(services->getPort("cca.GUIService"));
//     if (guiService.isNull()) {
//     }
//   }
//   catch (const sci::cca::CCAException::pointer &e) {
//     // error...
//   }
  wxMessageBox(com->getMessage(), "Hello Component", wxOK|wxICON_INFORMATION, 0);
#endif
  return 0;
}

int HelloGoPort::go()
{
    if (services.isNull()) {
        std::cerr << "Null services!\n";
        return 1;
    }
    std::cerr << "Hello.go.getPort...";
    double st = SCIRun::Time::currentSeconds();

    sci::cca::Port::pointer pp;
//     sci::cca::Port::pointer progPort;
    try {
        pp = services->getPort("stringport");
//         progPort = services->getPort("progress");
    }
    catch (const sci::cca::CCAException::pointer &e) {
        std::cerr << e->getNote() << std::endl;
        return 1;
    }

//      sci::cca::ports::Progress::pointer pPtr =
//          pidl_cast<sci::cca::ports::Progress::pointer>(progPort);

    sci::cca::ports::StringPort::pointer sp =
        pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
    std::string name = sp->getString();

    double t = Time::currentSeconds() - st;
    std::cerr << "Done in " << t << "secs\n";
    std::cerr << t*1000*1000 << " us/rep\n";

    if (! name.empty()) {
      com->setMessage(name);
    }

//     pPtr->updateProgress(HelloComponentIcon::STEPS);
    services->releasePort("stringport");
//     services->releasePort("progress");

    std::cout << " testing : " << com->getMessage() << std::endl;
    return 0;
}

// std::string HelloComponentIcon::getDisplayName()
// {
//     return std::string("Hello Component");
// }

// std::string HelloComponentIcon::getDescription()
// {
//     return std::string("The Hello component is a sample CCA component that uses a sci::cca::StringPort.");
// }

// int HelloComponentIcon::getProgressBar()
// {
//     return STEPS;
// }

// std::string HelloComponentIcon::getIconShape()
// {
//     return std::string("RECT");
// }


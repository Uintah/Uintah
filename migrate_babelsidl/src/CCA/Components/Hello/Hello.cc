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
#include <Framework/TypeMap.h>

#include <iostream>
#include <unistd.h>


using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Hello()
{
  return sci::cca::Component::pointer(new Hello());
}


Hello::Hello() : text("GO hasn't been called yet!"),
                 displayName("Hello Component"),
                 description("The Hello component is a sample CCA component that uses a sci::cca::StringPort.")
{
}

Hello::~Hello()
{
}

void Hello::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  services->registerForRelease(sci::cca::ComponentRelease::pointer(this));

  HelloUIPort *uip = new HelloUIPort(services);
  uip->setParent(this);
  HelloUIPort::pointer uiPortPtr = HelloUIPort::pointer(uip);

  services->addProvidesPort(uiPortPtr,
                            "ui",
                            "sci.cca.ports.UIPort",
                            sci::cca::TypeMap::pointer(0));

  services->addProvidesPort(sci::cca::ports::GoPort::pointer(this),
                            "go",
                            "sci.cca.ports.GoPort",
                            sci::cca::TypeMap::pointer(0));

  services->addProvidesPort(sci::cca::ports::ComponentIcon::pointer(this),
                            "icon",
                            "sci.cca.ports.ComponentIcon",
                            sci::cca::TypeMap::pointer(0));

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  props->putString("cca.portName", "stringport");
  props->putString("cca.portType", "sci.cca.ports.StringPort");

  services->registerUsesPort("stringport", "sci.cca.ports.StringPort", props);
}

void Hello::releaseServices(const sci::cca::Services::pointer& svc)
{
  services->unregisterUsesPort("stringport");

  services->removeProvidesPort("ui");
  services->removeProvidesPort("go");
  services->removeProvidesPort("icon");
}

int Hello::go()
{
  if (services.isNull()) {
    std::cerr << "Null services!\n";
    return -1;
  }
#if HAVE_GUI
  sci::cca::ports::GUIService::pointer guiService;
  try {
    guiService = pidl_cast<sci::cca::ports::GUIService::pointer>(services->getPort("cca.GUIService"));
    if (guiService.isNull()) {
      wxMessageBox(wxT("GUIService is not available"), wxT(getDisplayName()), wxOK|wxICON_ERROR, 0);
      return -2;
    }
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT(getDisplayName()), wxOK|wxICON_ERROR, 0);
  }
  sci::cca::ComponentID::pointer cid = services->getComponentID();
#endif

  double st = SCIRun::Time::currentSeconds();

#if HAVE_GUI
  guiService->updateProgress(cid, 20);
#endif

  sci::cca::Port::pointer pp;
  try {
    pp = services->getPort("stringport");
  }
  catch (const sci::cca::CCAException::pointer &e) {
#if HAVE_GUI
    wxMessageBox(e->getNote(), wxT(getDisplayName()), wxOK|wxICON_ERROR, 0);
#else
    std::cerr << e->getNote() << std::endl;
#endif
    return -1;
  }

#if HAVE_GUI
  guiService->updateProgress(cid, 50);
#endif

  sci::cca::ports::StringPort::pointer sp =
    pidl_cast<sci::cca::ports::StringPort::pointer>(pp);
  std::string name = sp->getString();

  double t = Time::currentSeconds() - st;
  std::cerr << "Done in " << t << "secs\n";
  std::cerr << t*1000*1000 << " us/rep\n";

  if (! name.empty()) {
    setMessage(name);
  }

#if HAVE_GUI
  guiService->updateProgress(cid, 100);
#endif

  services->releasePort("stringport");

#if HAVE_GUI
  services->releasePort("cca.GUIService");
#endif

  return 0;
}

int HelloUIPort::ui()
{
#if HAVE_GUI
  wxMessageBox(com->getMessage(), wxT(com->getDisplayName()), wxOK|wxICON_INFORMATION, 0);
#endif
  return 0;
}

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

#include <wx/splash.h>
#include <wx/bitmap.h>
#include <wx/image.h>

#include <CCA/Components/Builder/Builder.h>
#include <Core/Util/Environment.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>

//#include <iostream>

namespace GUIBuilder {

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_Builder()
{
  return sci::cca::Component::pointer(new Builder());
}

Mutex wxSCIRunApp::appLock("GUI application lock");
Semaphore wxSCIRunApp::sem("wxWidgets GUI Thread startup wait", 0);
sci::cca::BuilderComponent::pointer wxSCIRunApp::topBuilder(0);

const std::string Builder::guiThreadName("wxWidgets GUI Builder");
Mutex Builder::builderLock("Builder class lock");
wxSCIRunApp* Builder::app = 0;

// Don't automatically create main function (see wx/app.h).
// Initializes global application object.
IMPLEMENT_APP_NO_MAIN(wxSCIRunApp)


class wxGUIThread : public Runnable {
public:
    wxGUIThread(const sci::cca::BuilderComponent::pointer &bc) : builder(bc) {}
    virtual ~wxGUIThread() {}
    virtual void run();
    sci::cca::BuilderComponent::pointer getBuilder() { return builder; }

private:
   // store builder handle
    sci::cca::BuilderComponent::pointer builder;
};

class DestroyInstancesThread : public Runnable {
public:
    DestroyInstancesThread(const sci::cca::BuilderComponent::pointer &bc) : builder(bc) {}
    virtual ~DestroyInstancesThread() {}
    virtual void run();
    sci::cca::BuilderComponent::pointer getBuilder() { return builder; }

private:
    sci::cca::BuilderComponent::pointer builder;
};

void
wxGUIThread::run()
{
    std::cerr << "******************wxGUIThread::run()**********************" << std::endl;
    wxSCIRunApp::setTopBuilder(builder);
    int argc = 1;
    char *argv[1];
    argv[0] = "SCIRun2";
    // add framework URL to args???
    // initialize single wxApp instance and run main loop (Unix-specific):
    wxEntry(argc, argv);  // never returns, do initialization from OnInit
}

void
DestroyInstancesThread::run()
{
}

bool
wxSCIRunApp::OnInit()
{
  wxApp::OnInit(); // for command line processing (if any)

  Guard g(&appLock);
  Builder::setApp(*this);
  semUp();
  // TODO: get splash screen working

//     wxInitAllImageHandlers();

//     std::string path(sci_getenv("SCIRUN_SRCDIR"));
//     path += "/CCA/Components/GUIPrototypes/scirun2-splash.png";
//     wxBitmap bitmap(wxT(path), wxBITMAP_TYPE_PNG);
//     if (bitmap.Ok()) {
//         wxSplashScreen splash(bitmap,  wxSPLASH_TIMEOUT|wxSPLASH_CENTRE_ON_SCREEN, 6000, 0, -1, wxDefaultPosition, wxDefaultSize, wxFRAME_NO_TASKBAR|wxSTAY_ON_TOP);
//         splash.Show(true);
//     } else {
//       std::cerr << "bitmap not loaded" << std::endl;
//     }
//     wxYield();

  BuilderWindow *window = new BuilderWindow(topBuilder, 0);
  window->Show(true);
  SetTopWindow(window);

  return true;
}

void
wxSCIRunApp::addTopWindow(const sci::cca::BuilderComponent::pointer& bc)
{
  Guard g(&appLock);
std::cerr << "wxSCIRunApp::AddTopWindow(): from thread " << Thread::self()->getThreadName() << std::endl;

  // set the "main" top level window as parent
  wxWindow *top = GetTopWindow();
  BuilderWindow *window = new BuilderWindow(bc, top);
  window->Show(true);
}

Builder::Builder()
{
std::cerr << "Builder::Builder(): from thread " << Thread::self()->getThreadName() << std::endl;
}

Builder::~Builder()
{
std::cerr << "Builder::~Builder(): from thread " << Thread::self()->getThreadName() << std::endl;
}

// The first Builder to be instantiated (when wxTheApp is null) gets setServices
// run from the "main" thread.
// Subsequent Builders should only run in the GUI thread.
// wxSCIRunApp functions should only run in the GUI thread.
// (is is possible to defend against instantiating in the wrong thread?)
void
Builder::setServices(const sci::cca::Services::pointer &svc)
{
  Guard g(&builderLock);
  std::cerr << "Builder::setServices(..) from thread " << Thread::self()->getThreadName() << std::endl;
  services = svc;

  // What framework do we belong to?
  try {
    sci::cca::ports::FrameworkProperties::pointer fwkProperties = pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(services->getPort("cca.FrameworkProperties"));
    sci::cca::TypeMap::pointer tm = fwkProperties->getProperties();
    services->releasePort("cca.FrameworkProperties");
    frameworkURL = tm->getString("url", "NO URL AVAILABLE");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Framework URL is not available; " <<  e->getNote() << std::endl;
  }

  if (! wxTheApp) {
    Thread *t = new Thread(new wxGUIThread(sci::cca::BuilderComponent::pointer(this)), guiThreadName.c_str(), 0, Thread::NotActivated);
    t->setStackSize(8*256*1024);
    t->activate(false);
    t->detach();
    wxSCIRunApp::semDown();
  } else {
    if (Thread::self()->getThreadName() == guiThreadName) {
      app->addTopWindow(sci::cca::BuilderComponent::pointer(this));
    //} else {
    // add to event queue???
    }
  }
}

sci::cca::ComponentID::pointer
Builder::createInstance(const std::string& className, const sci::cca::TypeMap::pointer& properties)
{
  Guard g(&builderLock);
//std::cerr << "Builder::createInstance(): from thread " << Thread::self()->getThreadName() << std::endl;
  sci::cca::TypeMap::pointer tm = services->createTypeMap();
  sci::cca::ComponentID::pointer cid;
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    cid = bs->createInstance(std::string(), className, tm);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not create an instance of " << className << "; " <<  e->getNote() << std::endl;
  }
  return cid;
}

void
Builder::destroyInstance(const sci::cca::ComponentID::pointer& cid, float timeout)
{
  Guard g(&builderLock);
  std::string className = cid->getInstanceName();
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    bs->destroyInstance(cid, timeout);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not destroy an instance of " << className << "; " <<  e->getNote() << std::endl;
  }
}

int
Builder::destroyInstances(const SSIDL::array1<sci::cca::ComponentID::pointer>& cidArray, float timeout)
{
  Guard g(&builderLock);
  // do this in a new thread?
  std::string className;
  int destroyedCount = 0;
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));

    for (SSIDL::array1<sci::cca::ComponentID::pointer>::const_iterator iter = cidArray.begin(); iter != cidArray.end(); iter++) {
      className = (*iter)->getInstanceName();
      bs->destroyInstance(*iter, timeout);
      destroyedCount++;
    }
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not destroy an instance of " << className << "; " <<  e->getNote() << std::endl;
  }
  return destroyedCount;
}

void
Builder::getUsedPortNames(const sci::cca::ComponentID::pointer& cid, SSIDL::array1<std::string>& nameArray)
{
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    nameArray = bs->getUsedPortNames(cid);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not get uses ports for " << cid->getInstanceName() << "; " <<  e->getNote() << std::endl;
  }
}

void
Builder::getProvidedPortNames(const sci::cca::ComponentID::pointer& cid, SSIDL::array1<std::string>& nameArray)
{
  Guard g(&builderLock);
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    nameArray = bs->getProvidedPortNames(cid);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not get uses ports for " << cid->getInstanceName() << "; " <<  e->getNote() << std::endl;
  }
}

void
Builder::getComponentClassDescriptions(SSIDL::array1<sci::cca::ComponentClassDescription::pointer>& descArray)
{
  Guard g(&builderLock);
  try {
    sci::cca::ports::ComponentRepository::pointer rep =
      pidl_cast<sci::cca::ports::ComponentRepository::pointer>(services->getPort("cca.ComponentRepository"));

    descArray = rep->getAvailableComponentClasses();

    services->releasePort("cca.ComponentRepository");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not get component descriptions from component repository; " << e->getNote() << std::endl;
  }
}

void
Builder::getCompatiblePortList(const sci::cca::ComponentID::pointer &user,
			       const std::string& usesPortName,
			       const sci::cca::ComponentID::pointer &provider,
			       SSIDL::array1<std::string>& portArray)
{
  Guard g(&builderLock);
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    portArray = bs->getCompatiblePortList(user, usesPortName, provider);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not get compatible port list for " << usesPortName << "; " <<  e->getNote() << std::endl;
  }
}

bool Builder::go()
{
  //Guard g(&builderLock);
  return true;
}

void Builder::connectionActivity(const sci::cca::ports::ConnectionEvent::pointer& e)
{
  std::cerr << "Builder::connectionActivity(..)" << std::endl;
}

void Builder::componentActivity(const sci::cca::ports::ComponentEvent::pointer& e)
{
  std::cerr << "Builder::componentActivity: got event for component " << e->getComponentID()->getInstanceName() << std::endl;
}

}

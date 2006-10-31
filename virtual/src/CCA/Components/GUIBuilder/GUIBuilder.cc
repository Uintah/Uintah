/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/string.h>

#include <CCA/Components/GUIBuilder/GUIBuilder.h>
#include <CCA/Components/GUIBuilder/BuilderWindow.h>
#include <CCA/Components/GUIBuilder/ComponentIcon.h>
// for default namespace strings - these may be moved elsewhere in the future
#include <SCIRun/ComponentSkeletonWriter.h>

#include <sci_metacomponents.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/PortInstance.h>
#if HAVE_VTK
#include <SCIRun/Vtk/VtkPortInstance.h>
#endif
#include <SCIRun/CCA/CCAComponentModel.h>
#include <SCIRun/Internal/ApplicationLoader.h>

#include <Core/Util/Environment.h>
#include <Core/OS/Dir.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Guard.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/pidl_cast.h>

#ifndef DEBUG
#  define DEBUG 0
#endif

namespace GUIBuilder {

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_GUIBuilder()
{
  return sci::cca::Component::pointer(new GUIBuilder());
}

const std::string GUIBuilder::GUI_THREAD_NAME("wxWidgets GUI Builder");
const std::string GUIBuilder::DEFAULT_SRC_DIR(sci_getenv("SCIRUN_SRCDIR"));
const std::string GUIBuilder::DEFAULT_OBJ_DIR(sci_getenv("SCIRUN_OBJDIR"));
const std::string GUIBuilder::DEFAULT_CCA_COMP_DIR(GUIBuilder::DEFAULT_SRC_DIR + CCAComponentModel::DEFAULT_PATH);
const std::string GUIBuilder::GOPORT(ComponentSkeletonWriter::DEFAULT_SIDL_PORT_NAMESPACE + "GoPort");
const std::string GUIBuilder::UIPORT(ComponentSkeletonWriter::DEFAULT_SIDL_PORT_NAMESPACE + "UIPort");
const std::string GUIBuilder::PROGRESS_PORT(ComponentSkeletonWriter::DEFAULT_SIDL_PORT_NAMESPACE + "Progress");
const std::string GUIBuilder::COMPONENTICON_PORT(ComponentSkeletonWriter::DEFAULT_SIDL_PORT_NAMESPACE + "ComponentIcon");
const std::string GUIBuilder::APP_EXT_WILDCARD("(*." + ApplicationLoader::APP_EXT + ")");

Mutex GUIBuilder::builderLock("GUIBuilder class lock");
wxSCIRunApp* GUIBuilder::app = 0;

class wxGUIThread : public Runnable {
public:
    wxGUIThread(const sci::cca::GUIBuilder::pointer &bc) : builder(bc) {}
    virtual ~wxGUIThread() {}
    virtual void run();
    sci::cca::GUIBuilder::pointer getBuilder() { return builder; }

private:
   // store builder handle
    sci::cca::GUIBuilder::pointer builder;
};

// not implemented yet
class DestroyInstancesThread : public Runnable {
public:
    DestroyInstancesThread(const sci::cca::GUIBuilder::pointer &bc) : builder(bc) {}
    virtual ~DestroyInstancesThread() {}
    virtual void run();
    sci::cca::GUIBuilder::pointer getBuilder() { return builder; }

private:
    sci::cca::GUIBuilder::pointer builder;
};

void
wxGUIThread::run()
{
    std::cerr << "******************wxGUIThread::run()**********************" << std::endl;
    wxSCIRunApp::SetTopBuilder(builder);
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
  // implement
}

GUIBuilder::GUIBuilder()
{

#if DEBUG
  std::cerr << "GUIBuilder::GUIBuilder(): from thread " << Thread::self()->getThreadName() << std::endl;
#endif

 


}

GUIBuilder::~GUIBuilder()
{
#if DEBUG
  std::cerr << "GUIBuilder::~GUIBuilder(): from thread " << Thread::self()->getThreadName() << std::endl;
#endif
  try {
    // get framework properties
    sci::cca::ports::GUIService::pointer guiService =
      pidl_cast<sci::cca::ports::GUIService::pointer>(services->getPort("cca.GUIService"));
    guiService->removeBuilder(services->getComponentID()->getInstanceName());
    services->releasePort("cca.GUIService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: GUI service is not available; " <<  e->getNote() << std::endl;
  }
}

// The first GUIBuilder to be instantiated (when wxTheApp is null) gets setServices
// run from the "main" thread.
// Subsequent Builders should only run in the GUI thread.
// wxSCIRunApp functions should only run in the GUI thread.
// (is is possible to defend against instantiating in the wrong thread?)
void
GUIBuilder::setServices(const sci::cca::Services::pointer &svc)
{
  sci::cca::EventListener::pointer evptr(this);
  sci::cca::Port::pointer pp  = svc->getPort("cca.EventService");
  sci::cca::ports::EventService::pointer ptr =  pidl_cast<sci::cca::ports::EventService::pointer>(pp);
  if(ptr.isNull()){
    std::cout << "Pointer returned is Null!!!\n";
  }
  sci::cca::WildcardTopic::pointer topicPtr = ptr->createWildcardTopic("scirun2.services.builderservice.component.*");
  topicPtr->registerEventListener(std::string("GUIBuilder"),evptr);



#if DEBUG
  std::cerr << "GUIBuilder::setServices(..) from thread " << Thread::self()->getThreadName() << std::endl;
#endif
  builderLock.lock();
  services = svc;
  builderLock.unlock();

  sci::cca::TypeMap::pointer tm;
  // Get framework-specific properties
  try {
    // get framework properties
    sci::cca::ports::FrameworkProperties::pointer fwkProperties =
      pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(services->getPort("cca.FrameworkProperties"));
    tm = fwkProperties->getProperties();
    services->releasePort("cca.FrameworkProperties");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Framework properties service error (fatal); " <<  e->getNote() << std::endl;
    return;
  }
  frameworkURL = tm->getString("url", "NO URL AVAILABLE");
  std::string file = tm->getString("app file", "");
  // get config dir
  configDir = tm->getString("config dir", "");

  sci::cca::ports::ApplicationLoaderService::pointer appLoader;
  try {
    appLoader =
      pidl_cast<sci::cca::ports::ApplicationLoaderService::pointer>(services->getPort("cca.ApplicationLoaderService"));
    appLoader->setFileName(file);
    services->releasePort("cca.ApplicationLoaderService");
  } catch (const sci::cca::CCAException::pointer &pe) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    bw->DisplayErrorMessage("Error: application loader service error; " + pe->getNote());
    return;
  }

  if (! wxTheApp) {
    Thread *t = new Thread(new wxGUIThread(sci::cca::GUIBuilder::pointer(this)), GUI_THREAD_NAME.c_str(), 0, Thread::NotActivated);
    t->setStackSize(8*256*1024);
    t->activate(false);
    t->detach();
    wxSCIRunApp::semDown();
  } else {
#if DEBUG
    std::cerr << "GUIBuilder::setServices(..) try to add top window." << std::endl;
#endif
    if (Thread::self()->getThreadName() == GUI_THREAD_NAME) {
//       app->AddBuilder(sci::cca::GUIBuilder::pointer(this));
    } else {
      std::cerr << "Builders can only be created in the GUI thread...Aborting!" << std::endl;
      abort();
    }
  }

  try {
    // add this builder to the GUI service
    sci::cca::ports::GUIService::pointer guiService =
      pidl_cast<sci::cca::ports::GUIService::pointer>(services->getPort("cca.GUIService"));
    guiService->addBuilder(services->getComponentID()->getInstanceName(), sci::cca::GUIBuilder::pointer(this));
    services->releasePort("cca.GUIService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: GUI service is not available; " <<  e->getNote() << std::endl;
  }

  setDefaultPortColors();

  // register for scirun2.services.builderservice.component.* topics

}

//////////////////////////////////////////////////////////////////////////
// Mediator between wxWidgets GUI classes and cca.ComponentRepository.

void
GUIBuilder::getComponentClassDescriptions(SSIDL::array1<sci::cca::ComponentClassDescription::pointer>& descArray)
{
  try {
    sci::cca::ports::ComponentRepository::pointer rep =
      pidl_cast<sci::cca::ports::ComponentRepository::pointer>(services->getPort("cca.ComponentRepository"));

    descArray = rep->getAvailableComponentClasses();

    services->releasePort("cca.ComponentRepository");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not get component descriptions from component repository; " + e->getNote());
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Mediator between wxWidgets GUI classes and cca.BuilderService.

void
GUIBuilder::getPortInfo(const sci::cca::ComponentID::pointer& cid, const std::string& portName, std::string& model, std::string& type)
{
  sci::cca::TypeMap::pointer props;
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    props = bs->getPortProperties(cid, portName);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    std::cerr << "Error: Could not get port properties for " << portName << "; " <<  e->getNote() << std::endl;
  }

  model = props->getString(PortInstance::MODEL, "");
  type = props->getString(PortInstance::TYPE, "");
}

sci::cca::ComponentID::pointer
GUIBuilder::createInstance(const sci::cca::ComponentClassDescription::pointer& cd)
{
  sci::cca::ComponentID::pointer cid;
  try {
    sci::cca::TypeMap::pointer tm = services->createTypeMap();
    tm->putString("LOADER NAME", cd->getLoaderName());
    tm->putString("builder name", services->getComponentID()->getInstanceName());

    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    cid = bs->createInstance(std::string(), cd->getComponentClassName(), tm);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage(e->getNote());
    }
  }
  return cid;
}

void
GUIBuilder::destroyInstance(const sci::cca::ComponentID::pointer& cid, float timeout)
{
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
GUIBuilder::destroyInstances(const SSIDL::array1<sci::cca::ComponentID::pointer>& cidArray, float timeout)
{
  //Guard g(&builderLock);
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
GUIBuilder::getUsedPortNames(const sci::cca::ComponentID::pointer& cid, SSIDL::array1<std::string>& nameArray)
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
GUIBuilder::getProvidedPortNames(const sci::cca::ComponentID::pointer& cid, SSIDL::array1<std::string>& nameArray)
{
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    nameArray = bs->getProvidedPortNames(cid);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not get uses ports for " +
                              cid->getInstanceName() + "; " +  e->getNote());
    }
  }
}

void
GUIBuilder::getCompatiblePortList(const sci::cca::ComponentID::pointer &user,
			       const std::string& usesPortName,
			       const sci::cca::ComponentID::pointer &provider,
			       SSIDL::array1<std::string>& portArray)
{
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    portArray = bs->getCompatiblePortList(user, usesPortName, provider);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not get compatible port list for " +
                              usesPortName + "; " +  e->getNote());
    }
  }
}

// Bridge (SCIM) functions
void
GUIBuilder::getBridgeablePortList(const sci::cca::ComponentID::pointer &user,
                                  const std::string& usesPortName,
                                  const sci::cca::ComponentID::pointer &provider,
                                  SSIDL::array1<std::string>& portArray)
{
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    portArray = bs->getBridgeablePortList(user, usesPortName, provider);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not get compatible port list for " +
                              usesPortName + "; " +  e->getNote());
    }
  }
}

// Connecting components from different component models with a bridge:
// [user component] --- [bridge component] --- [provider component]
// where --- is a bridge connection
sci::cca::ComponentID::pointer
GUIBuilder::generateBridge(const sci::cca::ComponentID::pointer &user,
                           const std::string& usesPortName,
                           const sci::cca::ComponentID::pointer &provider,
                           const std::string& providesPortName,
                           sci::cca::ConnectionID::pointer& connID1,
                           sci::cca::ConnectionID::pointer& connID2)
{
  sci::cca::ComponentID::pointer bridgeCID;
#ifdef BUILD_BRIDGE

  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    std::string instanceName = bs->generateBridge(user, usesPortName, provider, providesPortName);
    if (instanceName.empty()) {
      BuilderWindow *bw = app->GetTopBuilderWindow();
      if (bw) {
        bw->DisplayErrorMessage("BuilderService was unable to generate bridge for " + usesPortName + " and " + providesPortName + ".");
      }
    } else {
      std::string className = "bridge:Bridge." + instanceName;
      sci::cca::TypeMap::pointer tm = services->createTypeMap();
      bridgeCID = bs->createInstance(instanceName, className, tm);

      // get all ports for newly created bridge component
      SSIDL::array1<std::string> usesPorts = bs->getUsedPortNames(bridgeCID);
      SSIDL::array1<std::string> providesPorts = bs->getProvidedPortNames(bridgeCID);
#if DEBUG
      std::cerr << "Bridge: connect " << user->getInstanceName() << "->"
                << usesPortName << " to " << bridgeCID->getInstanceName() << "->"
                << providesPorts[0] << std::endl;
#endif
      connID1 = bs->connect(user, usesPortName, bridgeCID, providesPorts[0]);
#if DEBUG
      std::cerr << "Bridge: connect " << bridgeCID->getInstanceName() << "->"
                << usesPorts[0] << " to " << provider->getInstanceName() << "->"
                << providesPortName << std::endl;
#endif
      connID2 = bs->connect(bridgeCID, usesPorts[0], provider, providesPortName);
    }
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not generate bridge for " + usesPortName + " and " + providesPortName + ";" +  e->getNote());
    }
  }
#endif
  return bridgeCID;
}


sci::cca::ConnectionID::pointer
GUIBuilder::connect(const sci::cca::ComponentID::pointer &usesCID, const std::string &usesPortName,
                 const sci::cca::ComponentID::pointer &providesCID, const ::std::string &providesPortName)
{
  sci::cca::ConnectionID::pointer connID;
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    connID = bs->connect(usesCID, usesPortName, providesCID, providesPortName);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not connect" + usesPortName + " to " +
			      providesPortName + "; " +  e->getNote());
    }
  }
  return connID;
}

void GUIBuilder::disconnect(const sci::cca::ConnectionID::pointer &connID, float timeout)
{
  try {
    sci::cca::ports::BuilderService::pointer bs =
      pidl_cast<sci::cca::ports::BuilderService::pointer>(services->getPort("cca.BuilderService"));
    bs->disconnect(connID, timeout);
    services->releasePort("cca.BuilderService");
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not disconnect; " +  e->getNote());
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Mediator between wxWidgets GUI classes and cca.FrameworkProxyService, cca.ComponentRepository.

// add component class described in XML file to the ComponentRepository at runtime
void GUIBuilder::addComponentFromXML(const std::string& filePath, const std::string& componentModel)
{
  BuilderWindow *bw = app->GetTopBuilderWindow();
  if (!bw) {
    std::cerr << "ERROR: could not get top builder window from wxSCIRunApp." << std::endl;
    return;
  }
  sci::cca::ports::FrameworkProperties::pointer fwkProperties;
  try {
    // append directory to sidl path!
    fwkProperties =
      pidl_cast<sci::cca::ports::FrameworkProperties::pointer>(services->getPort("cca.FrameworkProperties"));
  } catch (const sci::cca::CCAException::pointer &pe) {
    bw->DisplayErrorMessage("Error: framework properties not found; " + pe->getNote());
    return;
  }
  sci::cca::TypeMap::pointer tm = fwkProperties->getProperties();
  SSIDL::array1<std::string> sArray = tm->getStringArray("sidl_xml_path", sArray);
  sArray.push_back(filePath);
  tm->putStringArray("sidl_xml_path", sArray);

  services->releasePort("cca.FrameworkProperties");

  try {
    sci::cca::ports::ComponentRepository::pointer reg =
      pidl_cast<sci::cca::ports::ComponentRepository::pointer>(
                                                               services->getPort("cca.ComponentRepository"));

    reg->addComponentClass(componentModel);
  }
  catch(const sci::cca::CCAException::pointer &pe) {
    bw->DisplayErrorMessage("Error: component repository not found; " + pe->getNote());
    return;
  }
  services->releasePort("cca.ComponentRepository");

  // TODO: should update all active builder windows
  bw->BuildAllPackageMenus();
}

//////////////////////////////////////////////////////////////////////////
//  Mediator between wxWidgets GUI classes and cca.FrameworkProxyService

void GUIBuilder::addFrameworkProxy(const std::string &loaderName, const std::string &user, const std::string &domain, const std::string &loaderPath)
{
  std::cerr << "GUIBuilder::addFrameworkProxy(..): " << loaderName << ", " << user << ", " << domain << ", " << loaderPath << std::endl;
  sci::cca::ports::FrameworkProxyService::pointer fwkProxy;
  try {
    fwkProxy =
      pidl_cast<sci::cca::ports::FrameworkProxyService::pointer>(services->getPort("cca.FrameworkProxyService"));
    fwkProxy->addLoader(loaderName, user, domain, loaderPath);

    services->releasePort("cca.FrameworkProxyService");
  } catch (const sci::cca::CCAException::pointer &pe) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    bw->DisplayErrorMessage("Error: framework proxy service not found; " + pe->getNote());
    return;
  }
}

void GUIBuilder::removeFrameworkProxy(const std::string &loaderName)
{
  sci::cca::ports::FrameworkProxyService::pointer fwkProxy;
  try {
    fwkProxy =
      pidl_cast<sci::cca::ports::FrameworkProxyService::pointer>(services->getPort("cca.FrameworkProxyService"));
    fwkProxy->removeLoader(loaderName);

    services->releasePort("cca.FrameworkProxyService");
  } catch (const sci::cca::CCAException::pointer &pe) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    bw->DisplayErrorMessage("Error: framework proxy service not found; " + pe->getNote());
    return;
  }
}

//////////////////////////////////////////////////////////////////////////
// sci.cca.ports.GoPort support
//
// GUI may offer a 'go' menu item for a component icon corresponding to
// a GoPort provided by the component instance represented by the icon.
// To execute GoPorts's go function, the GUIBuilder component needs
// to use and connect the port.
// Modifies usesPortName: constructs unique (in scope of this GUIBuilder)
// uses port name for caller to store

bool GUIBuilder::connectGoPort(const std::string& usesName, const std::string& providesPortName,
                            const sci::cca::ComponentID::pointer &cid, std::string& usesPortName)
{
  usesPortName = usesName + "." + "goPort";
#if DEBUG
  std::cerr << "GUIBuilder::connectGoPort(..): uses port name=" << usesPortName
            << ", provides port name=" << providesPortName
            << ", component instance=" << cid->getInstanceName() << std::endl;
#endif
  // do we really need to look for SCIRun ports (ie. sci.go?)
  return connectPort(usesPortName, providesPortName, GOPORT, cid);
}

void GUIBuilder::disconnectGoPort(const std::string& goPortName)
{
#if DEBUG
  std::cerr << "GUIBuilder::disconnectGoPort(..): go port name=" << goPortName << std::endl;
#endif
  disconnectPort(goPortName);
}

int GUIBuilder::go(const std::string& goPortName)
{
  Guard g(&builderLock);
  sci::cca::ports::GoPort::pointer goPort;
  try {
    sci::cca::Port::pointer p = services->getPort(goPortName);
    goPort = pidl_cast<sci::cca::ports::GoPort::pointer>(p);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not access go port; " +  e->getNote());
    }
    return -1;
  }
  int status = goPort->go();

  services->releasePort(goPortName);
  return status;
}

//////////////////////////////////////////////////////////////////////////
// sci.cca.ports.UIPort support


bool GUIBuilder::connectUIPort(const std::string& usesName, const std::string& providesPortName, const sci::cca::ComponentID::pointer &cid, std::string& usesPortName)
{
  usesPortName = usesName + "." + "uiPort";
#if DEBUG
  std::cerr << "GUIBuilder::connectUIPort(..): uses port name=" << usesPortName
            << ", provides port name=" << providesPortName
            << ", component instance=" << cid->getInstanceName() << std::endl;
#endif
  // do we really need to look for SCIRun ports (ie. sci.ui?)
  return connectPort(usesPortName, providesPortName, UIPORT, cid);
}

void GUIBuilder::disconnectUIPort(const std::string& uiPortName)
{
#if DEBUG
  std::cerr << "GUIBuilder::disconnectUIPort(..): ui port name=" << uiPortName << std::endl;
#endif
  disconnectPort(uiPortName);
}

int GUIBuilder::ui(const std::string& uiPortName)
{
  Guard g(&builderLock);
  sci::cca::ports::UIPort::pointer uiPort;
  try {
    sci::cca::Port::pointer p = services->getPort(uiPortName);
    uiPort = pidl_cast<sci::cca::ports::UIPort::pointer>(p);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not access ui port; " +  e->getNote());
    }
    return -1;
  }
  int status = uiPort->ui();
  services->releasePort(uiPortName);
  return status;
}

/////////////////////////////////////////////////////////////////////////////
// component progress

void GUIBuilder::updateProgress(const sci::cca::ComponentID::pointer& cid, int progressPercent)
{
  // get all builder windows
  // this should be done in a worker thread
  BuilderWindow *bw = app->GetTopBuilderWindow();
  if (bw) {
    ComponentIcon* ci = bw->GetComponentIcon(cid->getInstanceName());
    if (ci) {
      ci->UpdateProgress(progressPercent);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// sci.cca.ports.ComponentIcon support

bool GUIBuilder::connectComponentIcon(const std::string& usesName, const std::string& providesPortName, const sci::cca::ComponentID::pointer &cid, std::string& usesPortName)
{
  usesPortName = usesName + "." + "componenticon";
#if DEBUG
  std::cerr << "GUIBuilder::connectComponentIcon(..): uses port name=" << usesPortName
            << ", provides port name=" << providesPortName
            << ", component instance=" << cid->getInstanceName() << std::endl;
#endif
  return connectPort(usesPortName, providesPortName, COMPONENTICON_PORT, cid);
}

void GUIBuilder::disconnectComponentIcon(const std::string& ciPortName)
{
#if DEBUG
  std::cerr << "GUIBuilder::disconnectComponentIcon(..): component icon port name=" << ciPortName << std::endl;
#endif
  disconnectPort(ciPortName);
}


//////////////////////////////////////////////////////////////////////////
// set port icon colors

bool GUIBuilder::setPortColor(const std::string& portType, const std::string& colorName)
{
  Guard g(&builderLock);
  wxColor c = wxTheColourDatabase->Find(wxT(colorName));
  if (! c.Ok()) {
    // colorName can't be found in wxTheColourDatabase
    return false;
  }

  for (PortColorMap::iterator iter = portColors.begin(); iter != portColors.end(); iter++) {
    if (iter->second == c) {
      return false;
    }
  }

  portColors[portType] = c;
  return true;
}

#if 0
// see Bugzilla bug #2834:
// void GUIBuilder::setPortColor(const std::string& portType, void* color)
// {
//   std::cerr << "GUIBuilder::setPortColor(..): portType=" << portType << std::endl;
//   Guard g(&builderLock);
//   wxColor c(*((wxColor*) color));
//   if (! c.Ok()) {
//     // bad color
//     return false;
//   }

//   for (PortColorMap::iterator iter = portColors.begin(); iter != portColors.end(); iter++) {
//     if (iter->second == c) {
//       return false;
//     }
//   }

//   portColors[portType] = c;
//   return true;
// }
#endif

void* GUIBuilder::getPortColor(const std::string& portType)
{
  Guard g(&builderLock);
  PortColorMap::iterator iter = portColors.find(portType);
  wxColor *c;
  if (iter == portColors.end()) {
    c = &(portColors[std::string("default")]);
  } else {
    c = &(iter->second);
  }

  return (void*) c;
}

// test ApplicationLoader

bool GUIBuilder::applicationFileExists()
{
  //Dir d();
  return false;
}

void GUIBuilder::saveApplication()
{
  sci::cca::ports::ApplicationLoaderService::pointer appLoader;
  try {
    appLoader =
      pidl_cast<sci::cca::ports::ApplicationLoaderService::pointer>(services->getPort("cca.ApplicationLoaderService"));
    appLoader->saveNetworkFile();
    services->releasePort("cca.ApplicationLoaderService");
  } catch (const sci::cca::CCAException::pointer &pe) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    bw->DisplayErrorMessage("Error: application loader service error; " + pe->getNote());
    return;
  }

}
// test ApplicationLoader


//////////////////////////////////////////////////////////////////////////
// CCA Event services
//


void GUIBuilder::processEvent(const std::string& topicName, const sci::cca::Event::pointer& theEvent)
{
  //process events
  sci::cca::TypeMap::pointer eventBody = theEvent->getBody();
  SSIDL::array1<std::string> allKeys = eventBody->getAllKeys(sci::cca::None);
  for (unsigned int i = 0; i < allKeys.size(); i++)
    std::cout << "Topic Name: " << topicName << "\tEvent Body: " << allKeys[i] << "\t" << eventBody->getString(allKeys[i],std::string("default")) << std::endl; 

}


///////////////////////////////////////////////////////////////////////////
// private member functions

void GUIBuilder::setDefaultPortColors()
{
  Guard g(&builderLock);

  // sci.cca.ports from SCIRun2Ports.sidl
  portColors[std::string("default")] = wxTheColourDatabase->Find(wxT("GOLD"));
  portColors[std::string("sci.cca.ports.StringPort")] = wxTheColourDatabase->Find(wxT("PINK"));
  portColors[std::string("sci.cca.ports.ZListPort")] = wxTheColourDatabase->Find(wxT("FIREBRICK"));
  portColors[std::string("sci.cca.ports.LinSolverPort")] = wxTheColourDatabase->Find(wxT("AQUAMARINE"));
  portColors[std::string("sci.cca.ports.PDEdescriptionPort")] = wxTheColourDatabase->Find(wxT("FOREST GREEN"));
  portColors[std::string("sci.cca.ports.MeshPort")] = wxTheColourDatabase->Find(wxT("ORANGE"));
  portColors[std::string("sci.cca.ports.ViewPort")] = wxTheColourDatabase->Find(wxT("PLUM"));
  portColors[std::string("sci.cca.ports.FEMmatrixPort")] = wxTheColourDatabase->Find(wxT("SLATE BLUE"));
  portColors[std::string("sci.cca.ports.BridgeTestPort")] = wxTheColourDatabase->Find(wxT("DARK GREY"));
  // VTK ports
#if HAVE_VTK
  portColors[VtkPortInstance::VTK_OUT_PORT] = wxTheColourDatabase->Find(wxT("SPRING GREEN"));
  portColors[VtkPortInstance::VTK_IN_PORT] = wxTheColourDatabase->Find(wxT("MEDIUM VIOLET RED"));
#endif
  // Babel ports?
}

bool GUIBuilder::connectPort(const std::string& usesPortName, const std::string& providesPortName, const std::string& portType, const sci::cca::ComponentID::pointer &cid)
{
  try {
    // have dialog to pack typemap? use XML file? set a preference?
    sci::cca::TypeMap::pointer tm = services->createTypeMap();
    services->registerUsesPort(usesPortName, portType, tm);
  } catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not register port " + usesPortName + "; " +  e->getNote());
    }
    return false;
  }

  sci::cca::ConnectionID::pointer connID =
    connect(services->getComponentID(), usesPortName, cid, providesPortName);
  if (connID.isNull()) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not connect port " + usesPortName + ".");
    }
    return false;
  }

  connectionMap[usesPortName] = connID;
  return true;
}

void GUIBuilder::disconnectPort(const std::string& usesPortName)
{
  //disconnect
  ConnectionMap::iterator iter = connectionMap.find(usesPortName);
  if (iter != connectionMap.end()) {
    disconnect(iter->second, 0);
    connectionMap.erase(iter);
  }

  try {
    services->unregisterUsesPort(usesPortName);
  } catch (const sci::cca::CCAException::pointer &e) {
    BuilderWindow *bw = app->GetTopBuilderWindow();
    if (bw) {
      bw->DisplayErrorMessage("Error: Could not unregister port " + usesPortName + "; " +  e->getNote());
    }
  }
}

}

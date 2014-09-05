#include <SCIRun/Internal/FrameworkProxyService.h>

#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/CCA/ComponentID.h>
#include <SCIRun/ComponentInstance.h>

namespace SCIRun {

FrameworkProxyService::FrameworkProxyService(SCIRunFramework* framework,
                   const std::string& name)
  : InternalComponentInstance(framework, name, "internal:FrameworkProxyService")
{
    this->framework=framework;
}

FrameworkProxyService::~FrameworkProxyService()
{
}

InternalComponentInstance*
FrameworkProxyService::create(SCIRunFramework* framework, const std::string& name)
{
    FrameworkProxyService* n = new FrameworkProxyService(framework, name);
    n->addReference();
    return n;
}

sci::cca::ComponentID::pointer
FrameworkProxyService::createInstance(const std::string& instanceName,
                   const std::string& className,
                   const sci::cca::TypeMap::pointer& properties)
{
    return framework->createComponentInstance(instanceName, className, properties);
}

int FrameworkProxyService::addLoader(const std::string &loaderName,
                              const std::string &user,
                              const std::string &domain,
                              const std::string &loaderPath)
{
    std::string sp = " ";
    std::cerr << "FrameworkProxyService::addLoader() not implemented" << std::endl;
    std::string cmd = "xterm -e ssh ";
    cmd+=user+"@"+domain+sp+loaderPath+sp+loaderName+sp+framework->getURL().getString() +"&";

    std::cout << cmd << std::endl;
    system(cmd.c_str());
    return 0;
}

int FrameworkProxyService::removeLoader(const std::string &loaderName)
{
    std::cerr << "FrameworkProxyService::removeLoader() not implemented" << std::endl;
    return 0;
}


int FrameworkProxyService::addComponentClasses(const std::string &loaderName)
{
    std::cerr<<"FrameworkProxyService::addComponentClasses not implemented" << std::endl;
    return 0;
}

int FrameworkProxyService::removeComponentClasses(const std::string &loaderName)
{
    std::cerr<<"FrameworkProxyService::removeComponentClasses not implemented" << std::endl;
    return 0;
}

sci::cca::Port::pointer
FrameworkProxyService::getService(const std::string&)
{
    return sci::cca::Port::pointer(this);
}


#if 0
/*
void FrameworkProxyService::registerFramework(const std::string &frameworkURL)
{
  Object::pointer obj=PIDL::objectFrom(frameworkURL);
  sci::cca::AbstractFramework::pointer remoteFramework=
    pidl_cast<sci::cca::AbstractFramework::pointer>(obj);
  sci::cca::Services::pointer bs = remoteFramework->getServices("external builder", 
                                "builder main", 
                                sci::cca::TypeMap::pointer(0));
  std::cerr << "got bs\n";
  sci::cca::ports::ComponentRepository::pointer reg =
    pidl_cast<sci::cca::ports::ComponentRepository::pointer>
    (bs->getPort("cca.ComponentRepository"));
  if (reg.isNull()) {
    std::cerr << "Cannot get component registry, not building component menus\n";
    return;
  }
  
  //traverse Builder Components here...

  for(unsigned int i=0; i<servicesList.size();i++) {
    sci::cca::ports::FrameworkProxyService::pointer builder 
      = pidl_cast<sci::cca::ports::FrameworkProxyService::pointer>
      (servicesList[i]->getPort("cca.FrameworkProxyService"));

    if (builder.isNull()) {
      std::cerr << "Fatal Error: Cannot find builder service\n";
      return;
    } 
    
    sci::cca::ComponentID::pointer cid=servicesList[i]->getComponentID();
    std::cerr<<"try to connect..."<<std::endl;
    sci::cca::ConnectionID::pointer connID=builder->connect(cid, "builderPort",
                                cid, "builder");
    std::cerr<<"connection done"<<std::endl;
  

    sci::cca::Port::pointer p = servicesList[i]->getPort("builder");
    sci::cca::ports::BuilderPort::pointer bp = 
      pidl_cast<sci::cca::ports::BuilderPort::pointer>(p);
    if (bp.isNull()) {
      std::cerr << "BuilderPort is not connected!\n";
    } 
    else{
      bp->buildRemotePackageMenus(reg, frameworkURL);
    }
    builder->disconnect(connID,0);
    servicesList[i]->releasePort("cca.FrameworkProxyService"); 
    servicesList[i]->releasePort("builder");
  }
  
}

void FrameworkProxyService::registerServices(const sci::cca::Services::pointer &svc)
{
  servicesList.push_back(svc);
}


sci::cca::AbstractFramework::pointer FrameworkProxyService::getFramework()
{
  return sci::cca::AbstractFramework::pointer(framework);
}
*/
#endif

}

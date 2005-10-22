#include <Core/CCA/spec/sci_sidl.h>
#include <main/SimpleManager.h>

#include <Core/Thread/Thread.h>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::ports;
  
  SimpleManager::SimpleManager(const AbstractFramework::pointer &framework, const std::string &appClass)
    : framework(framework), appClass(appClass)
  {
//     Thread *manager = new Thread( this, "SimpleManager");
//     manager->detach();
    run();
  }
  
  SimpleManager::~SimpleManager()
  {
  }
  
  void SimpleManager::run()
  {
    std::cout << "SimpleManger.\n";
    
    services = framework->getServices("test", "cca.unknown", 0);
    
    services->registerUsesPort("run", "sci.cca.ports.GoPort", 0);
    services->registerUsesPort("builder", "cca.BuilderService", 0);
    
    builder = pidl_cast<BuilderService::pointer>( services->getPort("builder"));
    
    ComponentID::pointer app = builder->createInstance("app", appClass, 0);
    ConnectionID::pointer connection = builder->connect(services->getComponentID(), "run", app, "go");
    GoPort::pointer run = pidl_cast<GoPort::pointer>(services->getPort("run"));
    
    run->go();
    
    services->releasePort("run");
    builder->disconnect(connection, 0);
    
    services->releasePort("builder");
    services->unregisterUsesPort("builder");
    services->unregisterUsesPort("run");
    
    framework->releaseServices(services);
  }
}

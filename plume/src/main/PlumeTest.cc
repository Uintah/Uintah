#include <Core/CCA/spec/sci_sidl.h>
#include <main/PlumeTest.h>

using namespace sci::cca;
using namespace sci::cca::ports;

PlumeTest::PlumeTest(const CoreFramework::pointer &framework)
  : framework(framework)
{
}

PlumeTest::~PlumeTest()
{
}

void PlumeTest::run()
{
  std::cout << "get services.\n";
  
  Services::pointer services = framework->getServices("test", "cca.unknown", framework->createTypeMap());
  
  services->registerUsesPort("test", "sci.cca.ports.GoPort", 0);
  services->registerUsesPort("builder", "cca.BuilderService", framework->createTypeMap());
  
  std::cout << "get builder.\n";
  BuilderService::pointer builder = pidl_cast<BuilderService::pointer>( services->getPort("builder"));
  
  // test
  std::cout << "create components\n";
  ComponentID::pointer hello = builder->createInstance("hello", "cca.core.Hello", 0);
  ComponentID::pointer world  = builder->createInstance("world", "cca.core.World", 0);
  
  std::cout << "connect components\n";
  ConnectionID::pointer connection = builder->connect( hello, "hello", world, "message");

  std::cout << "get Hello::go\n";
  ConnectionID::pointer goConnection = builder->connect(services->getComponentID(), "test", hello, "go");
  GoPort::pointer test = pidl_cast<GoPort::pointer>(services->getPort("test"));

  std::cout << "run\n\n";
  test->go();

  std::cout << "\ncleanup\n";
  services->releasePort("test");

  builder->disconnect( connection, 0 );
  builder->destroyInstance(hello, 0);
  builder->destroyInstance(world, 0);

  builder->disconnect(goConnection, 0);

  services->releasePort("builder");
  services->unregisterUsesPort("test");
  services->unregisterUsesPort("builder");
  framework->releaseServices(services);
  
  std::cout << "done\n";
}

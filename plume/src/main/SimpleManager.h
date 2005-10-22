#ifndef SimpleManager_h
#define SimpleManager_h

#include <Core/Thread/Runnable.h>

namespace SCIRun {

  using namespace sci::cca;
  using namespace sci::cca::ports;

  class SimpleManager : public Runnable 
  {
  public:
    SimpleManager(const AbstractFramework::pointer &, const std::string &className);
    virtual ~SimpleManager();
    
    virtual void run();
  protected:
    AbstractFramework::pointer framework;
    Services::pointer services;
    BuilderService::pointer builder;
    std::string appClass;
  };
}

#endif // SimpleManager_h

#ifndef SCIRun_PlumeFramework_h
#define SCIRun_PlumeFramework_h

#include <SCIRun/Distributed/DistributedFramework.h>

namespace SCIRun {

  class PlumeFramework : public DistributedFramework 
  {
  public:
    PlumeFramework(DistributedFramework::pointer &parent = 0);
    virtual ~PlumeFramework();

    CCAComponentModel cca;
    /*
     * Two pure virtual methods to create and destroy a component.
     */
    virtual ComponentInfo*
    createComponent( const std::string &, const std::string &, const sci::cca::TypeMap::pointer& properties);

    virtual void destroyComponent( const sci::cca::ComponentId::pointer &id);
  };

} // SCIRun namespace
    
#endif // SCIRun_PlumeFramework_h
 

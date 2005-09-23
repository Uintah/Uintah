#ifndef SCIRun_PlumeFramework_h
#define SCIRun_PlumeFramework_h

#include <SCIRun/Distributed/DistributedFramework.h>

namespace SCIRun {

  class PlumeFramework : public DistributedFramework 
  {
  public:
    PlumeFramework(DistributedFramework::pointer &parent = 0);
    virtual ~PlumeFramework();

    /*
     * Two pure virtual methods to create and destroy a component.
     */
    virtual sci::cca::Component::pointer 
    createComponent( const std::string &, const std::string &, const sci::cca::TypeMap::pointer& properties);

    virtual void destroyComponent( const sci::cca::ComponentID::pointer &);
  };

} // SCIRun namespace
    
#endif // SCIRun_PlumeFramework_h
 

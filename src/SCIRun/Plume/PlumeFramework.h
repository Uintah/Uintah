#ifndef SCIRun_PlumeFramework_h
#define SCIRun_PlumeFramework_h

#include <SCIRun/Plume/PlumeFrameworkImpl.h>

namespace SCIRun {

  class PlumeFramework : public PlumeFrameworkImpl<Plume::PlumeFramework>
  {
  public:
    typedef Plume::PlumeFramework::pointer pointer;

    PlumeFramework(const Distributed::internal::DistributedFrameworkInternal::pointer &parent
		   = Distributed::internal::DistributedFrameworkInternal::pointer(0) );
    virtual ~PlumeFramework();
  };

} // SCIRun namespace
    
#endif // SCIRun_PlumeFramework_h
 

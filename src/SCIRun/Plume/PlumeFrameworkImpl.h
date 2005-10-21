#ifndef SCIRun_Plume_PlumeFrameworkImpl_h
#define SCIRun_Plume_PlumeFrameworkImpl_h

#include <SCIRun/Plume/PlumeFrameworkBase.h>

namespace SCIRun {

  using namespace sci::cca;
  using namespace sci::cca::distributed;
  using namespace sci::cca::plume;

  class PlumeFrameworkImpl : public PlumeFrameworkBase<PlumeFramework>
  {
  public:
    typedef PlumeFramework::pointer pointer;

    PlumeFrameworkImpl(const DistributedFramework::pointer &parent = 0 );
    virtual ~PlumeFrameworkImpl();

    virtual AbstractFramework::pointer createEmptyFramework();
  };

} // SCIRun namespace
    
#endif // SCIRun_PlumeFramework_h
 

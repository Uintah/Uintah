
#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Plume/PlumeFrameworkImpl.h>
#include <SCIRun/Plume/PlumeFrameworkBase.code>

namespace SCIRun {

  using namespace sci::cca;
  using namespace sci::cca::core;
  using namespace sci::cca::distributed;
  using namespace sci::cca::plume;

  PlumeFrameworkImpl::PlumeFrameworkImpl( const DistributedFramework::pointer &parent )
    : PlumeFrameworkBase<PlumeFramework>(parent)
  {}

  PlumeFrameworkImpl::~PlumeFrameworkImpl() {}

  AbstractFramework::pointer PlumeFrameworkImpl::createEmptyFramework()
  { 
    return new PlumeFrameworkImpl(this); 
  }

} // namespace SCIRun

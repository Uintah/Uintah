#include <SCIRun/Plume/PlumeFramework.h>
#include <SCIRun/Plume/PlumeFrameworkImpl.code>
//#include <SCIRun/Distributed/DistributedFramework.h>

namespace SCIRun {

  PlumeFramework::PlumeFramework( const Distributed::internal::DistributedFrameworkInternal::pointer &parent )
    : PlumeFrameworkImpl<Plume::PlumeFramework>(parent)
  {}

  PlumeFramework::~PlumeFramework() {}

} // namespace SCIRun

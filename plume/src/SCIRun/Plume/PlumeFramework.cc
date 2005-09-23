#include <SCIRun/Plume/PlumeFramework.h>

namespace SCIRun {

  PlumeFramework::PlumeFramework( DistributedFramework::pointer &parent )
    : DistributedFramework(parent)
  {}

  PlumeFramework::~PlumeFramework() {}

  sci::cca::Component::pointer 
  PlumeFramework::createComponent( const std::string &instanceName, 
				   const std::string &className, 
				   const sci::cca::TypeMap::pointer& properties)
  {
    return sci::cca::Component::pointer(0);
  }

  void PlumeFramework::destroyComponent( const sci::cca::ComponentID::pointer &)
  {
  }

} // namespace SCIRun

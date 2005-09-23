#include <SCIRun/Plume/PlumeFramework.h>

namespace SCIRun {

  PlumeFramework::PlumeFramework( DistributedFramework::pointer &parent )
    : DistributedFramework(parent), cca(this)
  {}

  PlumeFramework::~PlumeFramework() {}

  ComponentInfo * 
  PlumeFramework::createComponent( const std::string &instanceName, 
				   const std::string &className, 
				   const sci::cca::TypeMap::pointer& properties)
  {
    return cca.createComponent( instanceName, className, properties );
  }

  void PlumeFramework::destroyComponent( const sci::cca::ComponentID::pointer &id)
  {
    ComponentInfo::pointer info = pidl_cast<ComponentInfo::pointer>(id);
    cca.destroyComponent(info.getPointer());
  }

} // namespace SCIRun

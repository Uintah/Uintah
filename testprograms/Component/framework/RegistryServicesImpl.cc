
#include <testprograms/Component/framework/RegistryServicesImpl.h>
#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/FrameworkImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>

#include <iostream>

namespace sci_cca {

using namespace std;

typedef Registry::component_iterator component_iterator;

RegistryServicesImpl::RegistryServicesImpl() 
{
}


RegistryServicesImpl::~RegistryServicesImpl()
{
}


void 
RegistryServicesImpl::init( const Framework &f ) 
{ 
  framework_ = f; 
  
  registry_ = dynamic_cast<FrameworkImpl *>(f.getPointer())->registry_;
}

void
RegistryServicesImpl::getActiveComponentList( array1<string> & components )
{
  registry_->connections_.readLock();
  
  Registry::component_iterator iter = registry_->components_.begin();

  for( ; iter != registry_->components_.end(); iter++ )
    {
      ComponentRecord * cr = (*iter).second;
      ComponentID & cid = cr->id_;
      components.push_back( cid->toString() );
    }

  registry_->connections_.readUnlock();
}

} // namespace sci_cca


#include <testprograms/Component/framework/RegistryServicesImpl.h>
#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/FrameworkImpl.h>

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
    cerr << "soon this will be a list of all active components\n";
    components.push_back( "a component" );
    components.push_back( "b component" );
    components.push_back( "c component" );
}

} // namespace sci_cca

#include <testprograms/Component/framework/ConnectionServicesImpl.h>
#include <testprograms/Component/framework/ComponentImpl.h>

#include <iostream>

using std::cerr;

namespace sci_cca {

ConnectionServicesImpl::ConnectionServicesImpl() 
{
}


ConnectionServicesImpl::~ConnectionServicesImpl()
{
}


void 
ConnectionServicesImpl::init( const Framework &f ) 
{ 
  framework_ = f; 
}


bool
ConnectionServicesImpl::connect( const ComponentID &, const string &, 
				 const ComponentID &, const string &)
{
}
  

bool 
ConnectionServicesImpl::disconnect( const ComponentID &, const string &, 
				    const ComponentID &, const string &)
{
}

bool 
ConnectionServicesImpl::exportAs( const ComponentID &, const string &, 
				  const string &)
{
}

bool
ConnectionServicesImpl::provideTo( const ComponentID &, const string&, 
				   const string &)
{
}

} // namespace sci_cca

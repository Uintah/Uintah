#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/ConnectionServicesImpl.h>
#include <testprograms/Component/framework/FrameworkImpl.h>

#include <iostream>

using std::cerr;

namespace sci_cca {

typedef Registry::component_iterator component_iterator;

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
  
  registry_ = dynamic_cast<FrameworkImpl *>(f.getPointer())->registry_;
}

bool
ConnectionServicesImpl::connect( const ComponentID &provide, 
				 const string &provide_port, 
				 const ComponentID &use, 
				 const string &use_port)
{
  component_iterator from = registry_->components_.find(provide);
  if ( from == registry_->components_.end() ) {
    // error: could not find provider
    return false; 
  }

  component_iterator to = registry_->components_.find(use);
  if ( to == registry_->components_.end() ) {
    // error: could not find user
    return false; 
  }

  ProvidePortRecord *ppr = from->second->getProvideRecord(provide_port);
  if ( !ppr ) {
    // error: could not find provider's port
    return false;
  }

  UsePortRecord *upr = from->second->getUseRecord(use_port);
  if ( !upr ) {
    // error: could not find use's port
    return false;
  }

  if ( ppr->connection_ ) {
    // error: provide port in use
    return false;
  }

  if ( upr->connection_ ) {
    // error: uses port in use
    return false;
  }

  ConnectionRecord *record = new ConnectionRecord;
  record->use_ = upr;
  record->provide_ = ppr;

  ppr->connection_ = record;
  upr->connection_ = record;

  // notify who ever wanted to 

  return true;
}
  

bool 
ConnectionServicesImpl::disconnect( const ComponentID &, const string &, 
				    const ComponentID &, const string &)
{
  return false;
}

bool 
ConnectionServicesImpl::exportAs( const ComponentID &, const string &, 
				  const string &)
{
  return false;
}

bool
ConnectionServicesImpl::provideTo( const ComponentID &, const string&, 
				   const string &)
{
  return false;
}

} // namespace sci_cca

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/BuilderServicesImpl.h>
#include <testprograms/Component/framework/FrameworkImpl.h>

#include <iostream>

namespace sci_cca {

using namespace std;

typedef Registry::component_iterator component_iterator;

BuilderServicesImpl::BuilderServicesImpl() 
{
}


BuilderServicesImpl::~BuilderServicesImpl()
{
}


void 
BuilderServicesImpl::init( const Framework::pointer &f ) 
{ 
  framework_ = f; 
  
  registry_ = dynamic_cast<FrameworkImpl *>(f.getPointer())->registry_;
}

bool
BuilderServicesImpl::connect( const ComponentID::pointer &uses, 
				 const string &use_port, 
				 const ComponentID::pointer &provider, 
				 const string &provide_port)
{
  // lock registry
  registry_->connections_.writeLock();

  // get provide port record
  ProvidePortRecord *provide = registry_->getProvideRecord( provider, 
							    provide_port );
  if ( !provide ) {
    // error: could not find provider's port
    cerr <<"provide not found\n";
    return false;
  }

  if ( provide->connection_ ) {
    // error: provide port in use
    cerr << "provide port in use\n";
    return false;
  }

  // get use port record
  cerr << "connections: uses id is " << uses->toString() << endl;
  UsePortRecord *use = registry_->getUseRecord( uses, use_port );

  if ( !use ) {
    // error: could not find use's port
    cerr << "uses not found\n";
    return false;
  }

  if ( use->connection_ ) {
    // error: uses port in use
    cerr << "use connection in use\n";
    return false;
  }

  // connect
  ConnectionRecord *record = new ConnectionRecord;
  record->use_ = use;
  record->provide_ = provide;

  provide->connection_ = record;
  use->connection_ = record;

  // unlock registry
  registry_->connections_.writeUnlock();

  // notify who ever wanted to 

  // done
  return true;
}
  

bool 
BuilderServicesImpl::disconnect( const ComponentID::pointer &, const string &, 
				    const ComponentID::pointer &, const string &)
{
  return false;
}

bool 
BuilderServicesImpl::exportAs( const ComponentID::pointer &, const string &, 
				  const string &)
{
  return false;
}

bool
BuilderServicesImpl::provideTo( const ComponentID::pointer &, const string&, 
				   const string &)
{
  return false;
}

} // namespace sci_cca

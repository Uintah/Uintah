
#include <testprograms/Component/framework/FrameworkImpl.h>
#include <testprograms/Component/framework/SciServicesImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/BuilderServicesImpl.h>
#include <testprograms/Component/framework/RegistryServicesImpl.h>
#include <testprograms/Component/framework/Registry.h>

#include <sys/utsname.h> 
#include <iostream>

namespace sci_cca {

using std::cerr;

FrameworkImpl::FrameworkImpl() 
  : ports_lock_("Framework Ports lock")
{

  // get hostname
  struct utsname rec;
  uname( &rec );
  hostname_ = rec.nodename;

  // create framework id
  ComponentIdImpl *id = new ComponentIdImpl;
  id->init( hostname_, "framework" );
  id_ = id;

  registry_ = new Registry;

  //
  // create services
  //

  // connection
  BuilderServicesImpl *csi = new BuilderServicesImpl;
  csi->init( Framework::pointer(this) );
  BuilderServices::pointer cs (csi);

  ports_["BuilderServices"] = cs;

  // Registry
  RegistryServicesImpl *rsi = new RegistryServicesImpl;

  rsi->init( Framework::pointer( this ) );
  RegistryServices::pointer rs( rsi );
  ports_["RegistryServices"] = rs;

  // directoty
  // creation
}

FrameworkImpl::~FrameworkImpl()
{
  cerr <<"FrameworkImpl destructor\n";
}

Port::pointer
FrameworkImpl::getPort( const ComponentID::pointer &id, const string &name )
{
  ports_lock_.readLock();

  // framework port ?
  port_iterator pi = ports_.find(name);
  if ( pi != ports_.end() ) {
    Port::pointer port = pi->second;
    ports_lock_.readUnlock();
    return port;
  }

  ports_lock_.readUnlock();

  // find in component record
  //  registry_->connections_.readLock();

  ComponentRecord *c = registry_->components_[id->toString()];
  Port::pointer port =  c->getPort( name );
  
  //registry_->connections_.readUnlock();

  return port;
}
    
void 
FrameworkImpl::registerUsesPort( const ComponentID::pointer &id, const PortInfo::pointer &info) 
{
  registry_->connections_.writeLock();

  ComponentRecord *c = registry_->components_[id->toString()];

  c->registerUsesPort( info );
  
  registry_->connections_.writeUnlock();
}


void 
FrameworkImpl::unregisterUsesPort( const ComponentID::pointer &id, const string &name )
{
  registry_->connections_.writeLock();
  
  ComponentRecord *c = registry_->components_[id->toString()];
  c->unregisterUsesPort( name );
  
  registry_->connections_.writeUnlock();
}

void 
FrameworkImpl::addProvidesPort( const ComponentID::pointer &id, const Port::pointer &port,
				const PortInfo::pointer& info) 
{
  registry_->connections_.writeLock();
  
  ComponentRecord *c = registry_->components_[id->toString()];
  c->addProvidesPort( port, info );
  
  registry_->connections_.writeUnlock();
}

void 
FrameworkImpl::removeProvidesPort( const ComponentID::pointer &id, const string &name)
{
  registry_->connections_.writeLock();
  
  ComponentRecord *c = registry_->components_[id->toString()];
  c->removeProvidesPort( name );
  
  registry_->connections_.writeUnlock();
}

void 
FrameworkImpl::releasePort( const ComponentID::pointer &id, const string &name)
{
  ports_lock_.readLock();
  bool found = ports_.find(name) != ports_.end();
  ports_lock_.readUnlock();

  if ( !found ) {
    registry_->connections_.writeLock();

    ComponentRecord *c = registry_->components_[id->toString()];
    c->releasePort( name );

    registry_->connections_.writeUnlock();
  }
}

bool
FrameworkImpl::registerComponent( const string &hostname, 
				  const string &program,
				  Component::pointer &c )
{
  // create new ID and Services objects
  ComponentIdImpl *id = new ComponentIdImpl;
  id->init(hostname, program );
  ComponentID::pointer cid(id);

  cerr << "framework register " << id->toString() << "\n";

  SciServices::pointer svc(new SciServicesImpl);
  svc->init(Framework::pointer(this), cid); 

  // save info about the Component 
  ComponentRecord *cr = new ComponentRecord( cid );
  cr->component_ = c;
  cr->services_ = svc;

//  map<string,ComponentRecord*>::iterator i; 
//   cerr << "Framework::before...\n";
//   for ( i=registry_->components_.begin(); i!=registry_->components_.end(); i++)
//     cerr << "component " << i->first << " -> " 
// 	 << i->second->id_->toString() << endl;

  registry_->connections_.writeLock();
  registry_->components_[id->toString()] = cr;
  registry_->connections_.writeUnlock();

//   cerr << "Framework::after...\n";
//   for ( i=registry_->components_.begin(); i!=registry_->components_.end(); i++)
//     cerr << "component " << i->first << " -> " 
// 	 << i->second->id_->toString() << endl;

  // initialized component
  c->setServices( svc );
  
  return true;
}

void
FrameworkImpl::unregisterComponent( const ComponentID::pointer &id )
{
  registry_->connections_.writeLock();

  // find the record
  ComponentRecord *cr = registry_->components_[id->toString()];

  // erase the entry
  registry_->components_.erase( id->toString() );

  registry_->connections_.writeUnlock();


  // explicitly erase the record
  delete cr;

}

void
FrameworkImpl::shutdown()
{
  // clear registry
  registry_->connections_.readLock();
  
  Registry::component_iterator iter = registry_->components_.begin();

  for( ; iter != registry_->components_.end(); iter++ ) {
    ComponentRecord * cr = (*iter).second;
    
    cr->component_->setServices( Services::pointer(0) );
    cr->component_ = 0;
    cr->services_ = 0;
    cr->id_ = 0;
  }
  registry_->connections_.readUnlock();

  // remove framework ports
  ports_.clear();
}

} // namespace sci_cca




#include <sys/utsname.h> 

#include <testprograms/Component/framework/SciServicesImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/ConnectionServicesImpl.h>
#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/FrameworkImpl.h>


namespace sci_cca {


FrameworkImpl::FrameworkImpl() 
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
  ConnectionServicesImpl *csi = new ConnectionServicesImpl;
  csi->init( Framework(this) );
  ConnectionServices cs = csi; 

  ports_["ConnectionServices"] = cs;

  // repository
  // directoty
  // creation
}

FrameworkImpl::~FrameworkImpl()
{
}

Port
FrameworkImpl::getPort( const ComponentID &id, const string &name )
{
  // framework port ?
  port_iterator pi = ports_.find(name);
  if ( pi != ports_.end() )
    return pi->second;

  // find in component record
  ComponentRecord *c = registry_->components_[id];
  return c->getPort( name );
}
    
void 
FrameworkImpl::registerUsesPort( const ComponentID &id, const PortInfo &info) 
{
  ComponentRecord *c = registry_->components_[id];
  c->registerUsesPort( info );
}


void 
FrameworkImpl::unregisterUsesPort( const ComponentID &id, const string &name )
{
  ComponentRecord *c = registry_->components_[id];
  c->unregisterUsesPort( name );
}

void 
FrameworkImpl::addProvidesPort( const ComponentID &id, const Port &port,
				const PortInfo& info) 
{
  ComponentRecord *c = registry_->components_[id];
  c->addProvidesPort( port, info );
}

void 
FrameworkImpl::removeProvidesPort( const ComponentID &id, const string &name)
{
  ComponentRecord *c = registry_->components_[id];
  c->removeProvidesPort( name );
}

void 
FrameworkImpl::releasePort( const ComponentID &id, const string &name)
{
  if ( ports_.find(name) != ports_.end() ) {
    // nothing to do for framework ports
  } else {
    ComponentRecord *c = registry_->components_[id];
    c->releasePort( name );
  }
}

bool
FrameworkImpl::registerComponent( const string &hostname, 
				  const string &program,
				  Component &c )
{
  // create new ID and Services objects
  ComponentIdImpl *id = new ComponentIdImpl;
  id->init(hostname, program );

  cerr << "framework register " << id->toString() << "\n";

  SciServices svc = new SciServicesImpl;
  svc->init(Framework(this), id); 

  // save info about the Component 
  ComponentRecord *cr = new ComponentRecord( id );
  cr->component_ = c;
  cr->services_ = svc;
  registry_->components_[id] = cr;

  // initialized component
  c->setServices( svc );
  
  return true;
}

void
FrameworkImpl::unregisterComponent( const ComponentID &id )
{
  cerr << "framework::unregister \n";

  // find the record
  ComponentRecord *cr = registry_->components_[id];

  // explicitly erase it
  delete cr;

  // erase the entry
  registry_->components_.erase( id );
}


} // namespace sci_cca



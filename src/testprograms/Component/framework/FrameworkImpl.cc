
#include <testprograms/Component/framework/SciServicesImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/ConnectionServicesImpl.h>
#include <testprograms/Component/framework/FrameworkImpl.h>


namespace sci_cca {

string make_fullname( const ComponentID &id, const string &name )
{
  return id->toString()+"/"+name;
}

class ComponentRecord {
public:
  Component component_;
  Services services_;
};

class ConnectionRecord;

class PortRecord {
public:
  ComponentID id_;
  PortInfo info_;
  ConnectionRecord *connection_;
};

class UsePortRecord : public PortRecord {
public:
};

class ProvidePortRecord : public PortRecord {
public:
  Port port_;
  bool in_use_;
};

class ConnectionRecord {
public:
  void disconnect();

public:
  UsePortRecord *use_;
  ProvidePortRecord *provide_;
};


void
ConnectionRecord::disconnect()
{
  // disconnect connection
  
  // clean up
  use_->connection_ = 0;
  provide_->connection_ = 0;
  delete this;
}

FrameworkImpl::FrameworkImpl()
{
  // create services

  // connection
  ConnectionServicesImpl *csi = new ConnectionServicesImpl;
  csi->init( Framework(this) );
  ConnectionServices cs = csi; 
  framework_ports_["ConnectionServices"] = cs;

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
  // is it a framework port ?
  framework_map::iterator p = framework_ports_.find(name);
  if ( p  != framework_ports_.end() ) 
    return p->second; 
  
  // no.
  use_map::iterator u = use_ports_.find(make_fullname(id,name));
  if ( u == use_ports_.end() ) {
    // throw an error ? the port was not registered as a use port.
    return 0;
  }

  UsePortRecord *use = u->second;
  
  if ( !use->connection_ )
    // a non blocking implementation
    return 0;
  
  ProvidePortRecord *provide = use->connection_->provide_;
  if ( !provide ) 
    throw InternalError( "Provide port is missing..." );

  return provide->port_;
}
    
void 
FrameworkImpl::registerUsesPort( const ComponentID &id, const PortInfo &info) 
{
  UsePortRecord *record = new UsePortRecord;
  record->id_ = id;
  record->info_ = info;
  record->connection_ = 0;

  use_ports_[make_fullname(id,info->getName())] = record;
}


void 
FrameworkImpl::unregisterUsesPort( const ComponentID &id, const string &name )
{
  string fullname = make_fullname(id,name);
  
  use_map::iterator use = use_ports_.find( fullname);
  if ( use != use_ports_.end() ) {
    // check if the use port is connected. 
    if ( use->second->connection_ ) {
      // if so than disconnect it
      use->second->connection_->disconnect();
    }
    delete use->second;
    use_ports_.erase(use);
  }
  else {
    // should it report an error if port was not found ?
  }
}

void 
FrameworkImpl::addProvidesPort( const ComponentID &id, const Port &port,
				const PortInfo& info) 
{
  ProvidePortRecord *record = new ProvidePortRecord;
  record->id_ = id;
  record->info_ = info;
  record->connection_ = 0;
  record->port_ = port;
  record->in_use_ = false;

  provide_ports_[make_fullname(id,info->getName())] = record;
}

void 
FrameworkImpl::removeProvidesPort( const ComponentID &id, const string &name)
{
  string fullname = make_fullname(id,name);
  provide_map::iterator p = provide_ports_.find(fullname);
  if ( p != provide_ports_.end() ) {
    if ( p->second->connection_ ) {
      p->second->connection_->disconnect();
    }
    delete p->second;
    provide_ports_.erase(p);
  }
  else {
    // should it report an error if port was not found ?
  }
}

void 
FrameworkImpl::releasePort( const ComponentID &id, const string &name)
{
  if ( framework_ports_.find( name ) != framework_ports_.end() ) {
    cerr << "framework port '"<<name<<"' released\n"; 
    // nothing to do for a framework port
    return;
  }

  string fullname = make_fullname(id,name);
  use_map::iterator use = use_ports_.find(fullname);
  if ( use == use_ports_.end() ) {
    throw InternalError( "Release port does not exist" );
  }

  UsePortRecord *record = use->second;

  if ( record->connection_ ) {
    record->connection_->disconnect();
    record->connection_ = 0;
  } else {
    // is this an error worthy of an exception
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
  ComponentRecord *cr = new ComponentRecord;
  cr->component_ = c;
  cr->services_ = svc;
  components_[id] = cr;

  // initialized component
  c->setServices( svc );
  
  return true;
}

void
FrameworkImpl::unregisterComponent( const ComponentID &id )
{
  cerr << "framework::unregister \n";

  // find the record
  ComponentRecord *cr = components_[id];

  // explicitly erase it
  delete cr;

  // erase the entry
  components_.erase( id );
}


} // namespace sci_cca



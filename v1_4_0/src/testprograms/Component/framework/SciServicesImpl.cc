#include <testprograms/Component/framework/SciServicesImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <iostream>

using std::cerr;

namespace sci_cca {

SciServicesImpl::SciServicesImpl() 
{
}

SciServicesImpl::~SciServicesImpl()
{
  cerr << "SciServices " << id_.getPointer() << " exit\n";
}

Port::pointer
SciServicesImpl::getPort( const string &name )
{
  return framework_->getPort( id_, name );
}

PortInfo::pointer
SciServicesImpl::createPortInfo( const string &name,
			      const string &type,
			      const array1<string> &properties )
{
  return PortInfo::pointer(new PortInfoImpl( name, type, properties ));
}

void
SciServicesImpl::registerUsesPort( const PortInfo::pointer &port_info )
{
  framework_->registerUsesPort( id_, port_info );
}

void
SciServicesImpl::unregisterUsesPort( const string &name )
{
  framework_->unregisterUsesPort( id_, name );
}

void   
SciServicesImpl::addProvidesPort( const Port::pointer &port,
				  const PortInfo::pointer &info )
{
  framework_->addProvidesPort( id_, port, info );
}

void
SciServicesImpl::removeProvidesPort( const string & name )
{
  framework_->removeProvidesPort( id_, name );
}

void
SciServicesImpl::releasePort( const string & name )
{
  framework_->releasePort( id_, name );
}
 
ComponentID::pointer
SciServicesImpl::getComponentID()
{
  return id_;
}

void 
SciServicesImpl::init( const Framework::pointer &f,
		       const ComponentID::pointer & id) 
{ 
  framework_ = f; 
  id_ = id; 
}

void
SciServicesImpl::done()
{
  framework_ = 0;
  id_ = 0;
}

void 
SciServicesImpl::shutdown() 
{ 
  framework_->unregisterComponent( id_); 
}

} // namespace sci_cca

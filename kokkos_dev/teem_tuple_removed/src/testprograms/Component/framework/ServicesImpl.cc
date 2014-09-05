#include <testprograms/Component/framework/ServicesImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <iostream>

using std::cerr;

namespace sci_cca {

ServicesImpl::ServicesImpl() 
{
}


ServicesImpl::~ServicesImpl()
{
  cerr << "Services " << id_ << " exit\n";
}

Port
ServicesImpl::getPort( const string & name )
{
  return 0;
}

PortInfo
ServicesImpl::createPortInfo( const string & name,
			      const string & type,
			      const array1<string> & properties )
{
  return new PortInfoImpl( name, type, properties );
}

void
ServicesImpl::registerUsesPort( const PortInfo & nameAndType )
{
}

void
ServicesImpl::unregisterUsesPort( const string & name )
{
}

void   
ServicesImpl::addProvidesPort( const Port & inPort,
		 	       const PortInfo & name )
{
  cerr << "ServicesImpl: addProvidesPort  not implemented yet\n";
//   throw InternalError( (string("addProvidesPort for: ") 
// 		       + " is not implemented yet.").c_str());

}

void
ServicesImpl::removeProvidesPort( const string & name )
{
  cerr << "removeProvidesPort not implemented yet\n";
  throw InternalError((string("removeProvidesPort for: ") + name
		       + " is not implemented yet.").c_str());
}

void
ServicesImpl::releasePort( const string & name )
{
  cerr << "releasePort not implemented yet\n";
  throw InternalError( (string("releasePort for: ") + name
			+ " is not implemented yet.").c_str() );
}
 
ComponentID
ServicesImpl::getComponentID()
{
  return id_;
}


} // namespace sci_cca

#include <testprograms/Component/framework/PortInfoImpl.h>

namespace sci_cca {

PortInfoImpl::PortInfoImpl( const string & name, 
			    const string & type, 
			    const array1<string> & properties )
  : name_(name), type_(type), properties_(properties)
{
}

PortInfoImpl::~PortInfoImpl()
{
}

string
PortInfoImpl::getType()
{
  return type_;
}

string
PortInfoImpl::getName()
{
  return name_;
}

string
PortInfoImpl::getProperty( const string & name )
{
  // Dd: Why is this +2?
  for( unsigned int i=0; i < properties_.size(); i += 2 ) {
    if( properties_[i] == name ) 
      return properties_[i+1];
  }

  return "";
}

} // namespace sci_cca

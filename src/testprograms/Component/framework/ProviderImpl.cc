
#include <testprograms/Component/framework/TestPortImpl.h>
#include <testprograms/Component/framework/ProviderImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <sstream>
#include <iostream>

namespace sci_cca {

using std::cerr;


ProviderImpl::ProviderImpl()
{
  test_port_ = new TestPortImpl;
}

ProviderImpl::~ProviderImpl()
{
}

void 
ProviderImpl::setServices( const Services &svc )
{
  ComponentImpl::setServices( svc );

  PortInfo info = new PortInfoImpl( "Provides", "", 0 );
  svc->addProvidesPort( test_port_, info );
}

} // namespace sci_cca


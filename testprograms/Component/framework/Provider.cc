
#include <testprograms/Component/framework/Provider.h>

#include <testprograms/Component/framework/TestPortImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <sstream>
#include <iostream>

namespace sci_cca {

using std::cerr;

Provider::Provider()
{
  test_port_ = new TestPortImpl;
}

Provider::~Provider()
{
}

void 
Provider::setServices( const Services &svc )
{
  ComponentImpl::setServices( svc );

  if( svc )
    {
      PortInfo info = new PortInfoImpl( "Provides", "", 0 );
      svc->addProvidesPort( test_port_, info );
    }
}

} // namespace sci_cca


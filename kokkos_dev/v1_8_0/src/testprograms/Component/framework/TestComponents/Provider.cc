
#include <testprograms/Component/framework/TestComponents/Provider.h>

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
Provider::setServices( const Services::pointer &svc )
{
  ComponentImpl::setServices( svc );

  if( !svc.isNull() )
    {
      PortInfo::pointer info (new PortInfoImpl( "Provides", "", 0 ));
      svc->addProvidesPort( test_port_, info );
    }
}

} // namespace sci_cca


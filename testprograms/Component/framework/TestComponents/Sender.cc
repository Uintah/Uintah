
#include <testprograms/Component/framework/TestComponents/Sender.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <sstream>
#include <iostream>

namespace sci_cca {

using std::cerr;

Sender::Sender()
{
}

Sender::~Sender()
{
  cout << "~Sender()\n";
}

void 
Sender::setServices( const Services &svc )
{
  ComponentImpl::setServices( svc );

  if( svc )
    {
      PortInfo info = new PortInfoImpl("Uses", "", 0);
      svc->registerUsesPort( info );
    }
}

void
Sender::go()
{
  TestPort port = pidl_cast<TestPort>(services_->getPort("Uses"));
  if ( !port ) {
    cerr << "go: could not find test port\n";
    return;
  }

  port->test();
}

} // namespace sci_cca


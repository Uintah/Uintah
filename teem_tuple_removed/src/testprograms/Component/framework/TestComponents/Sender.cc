
#include <testprograms/Component/framework/TestComponents/Sender.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <sstream>
#include <iostream>
using namespace std;

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
Sender::setServices( const Services::pointer &svc )
{
  ComponentImpl::setServices( svc );

  if( !svc.isNull() )
    {
      PortInfo::pointer info (new PortInfoImpl("Uses", "", 0));
      svc->registerUsesPort( info );
    }
}

void
Sender::go()
{
  TestPort::pointer port (pidl_cast<TestPort::pointer>(services_->getPort("Uses")));
  if ( port.isNull() ) {
    cerr << "go: could not find test port\n";
    return;
  }

  port->test();
}

} // namespace sci_cca


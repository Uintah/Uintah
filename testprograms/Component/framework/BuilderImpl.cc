
#include <testprograms/Component/framework/BuilderImpl.h>

#include <sstream>

namespace sci_cca {

BuilderImpl::BuilderImpl()
{
  cerr << "Builder::" << endl; 
}

BuilderImpl::~BuilderImpl()
{
}

void 
BuilderImpl::setServices( const Services &svc )
{
  cerr << "Builder set serevices\n";
  ComponentImpl::setServices( svc );

  if ( svc ) {
    Port port = svc->getPort("ConnectionServices");
    if ( !port ) {
      cerr << "Could not get connection port\n";
    }
    else {
      cerr << "releasing connection port..\n";
      svc->releasePort( "ConnectionServices");
    }
  }
  cerr << "Builder set services done\n";

}

} // namespace sci_cca


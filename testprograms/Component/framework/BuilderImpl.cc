
#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/BuilderImpl.h>
#include <testprograms/Component/framework/SenderImpl.h>
#include <testprograms/Component/framework/ProviderImpl.h>

#include <sstream>
#include <iostream>

namespace sci_cca {

using std::cerr;

BuilderImpl::BuilderImpl()
{
}

BuilderImpl::~BuilderImpl()
{
}

void 
BuilderImpl::setServices( const Services &svc )
{
  ComponentImpl::setServices( svc );

  if ( svc ) {
    ConnectionServices port = pidl_cast<ConnectionServices>(
                                          svc->getPort("ConnectionServices"));
    if ( !port ) {
      cerr << "Could not get connection port\n";
      return;
    }

#if 0
    RegistryServices reg_port = pidl_cast<RegistryServices>(
                                           svc->getPort("RegistryServices"));
    if ( !reg_port ) {
      cerr << "Could not get registry port\n";
      return;
    } else {
      array1<string> components;
      reg_port->getActiveComponentList( components );

      cerr << components.size() << " components returned:\n";

      for( int cnt = 0; cnt < components.size(); cnt++ )
	{
	  cerr << cnt << ": " << components[ cnt ] << "\n";
	}
    }
#endif

    Sender sender = new SenderImpl;
    Component s = sender;
    CCA::init( s );
    ComponentID sid = sender->getComponentID();
  
    Provider provider = new ProviderImpl;
    Component p = provider;
    CCA::init( p );
    ComponentID pid = provider->getComponentID();
    
    port->connect( sid, "Uses", pid, "Provides");
    
    sender->go();
    
    svc->releasePort( "ConnectionServices");
  }
}

} // namespace sci_cca


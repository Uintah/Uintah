
#include <testprograms/Component/framework/Builder.h>

#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/Sender.h>
#include <testprograms/Component/framework/Provider.h>

#include <sstream>
#include <iostream>

namespace sci_cca {

using std::cerr;

Builder::Builder()
{
}

Builder::~Builder()
{
}

void 
Builder::setServices( const Services &svc )
{
  ComponentImpl::setServices( svc );

  if ( svc ) {
    ConnectionServices connect_port = pidl_cast<ConnectionServices>(
                                          svc->getPort("ConnectionServices"));
    if ( !connect_port ) {
      cerr << "Could not get connection port\n";
      return;
    }

    //////////////////////////////////////////////////////////////////////
    // Query the Registry for all active components...
    RegistryServices reg_port = pidl_cast<RegistryServices>(
                                           svc->getPort("RegistryServices"));
    if ( !reg_port ) {
      cerr << "Could not get registry port\n";
      return;
    } else {
      array1<string> components;
      reg_port->getActiveComponentList( components );

      cerr << components.size() << " components returned:\n";

      for( unsigned int cnt = 0; cnt < components.size(); cnt++ )
	{
	  cerr << cnt << ": " << components[ cnt ] << "\n";
	}
      svc->releasePort( "RegistryServices" );
    }
    //////////////////////////////////////////////////////////////////////

    // Testing: Creating and connecting two generic components...
    Sender * sender = new Sender();
    Component s = sender;
    CCA::init( s );
    ComponentID sid = sender->getComponentID();
  
    Provider * provider = new Provider();
    Component p = provider;
    CCA::init( p );
    ComponentID pid = provider->getComponentID();
    
    connect_port->connect( sid, "Uses", pid, "Provides" );
    
    sender->go();
    
    svc->releasePort( "ConnectionServices" );
  }
}

} // namespace sci_cca


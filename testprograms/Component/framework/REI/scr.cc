
#include <testprograms/Component/framework/REI/scr.h>

using namespace sci_cca;
#include <iostream>
using namespace std;

void
scr::setServices( const Services & srv )
{
  ComponentImpl::setServices( srv );

  if( srv )
    {
      cout << "Registering scrOut0 provides port\n";

      PortInfo info = srv->createPortInfo( "scrOut0", 
					   "sci_cca.scrInterface", 0 );

      scr_port_ = new scrInterfaceImpl();

      scrInterface scrPort = scr_port_;

      srv->addProvidesPort( scrPort, info );
    }
}

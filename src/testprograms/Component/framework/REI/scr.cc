
#include <testprograms/Component/framework/REI/scr.h>

using namespace sci_cca;

void
scr::setServices( const Services & srv )
{
  if( srv )
    {
      PortInfo info = srv->createPortInfo( "scrPort0", "scrInterface", 0 );

      scr_port_ = new scrInterfaceImpl();

      scrInterface scrPort = scr_port_;

      srv->addProvidesPort( scrPort, info );
    }
}

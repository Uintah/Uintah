
#include <testprograms/Component/framework/scr.h>

using namespace sci_cca;

void
scr::setServices( const Services & srv )
{

  scr_port_ = new scrInterfaceImpl();

}

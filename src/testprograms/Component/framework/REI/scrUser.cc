
#include <testprograms/Component/framework/REI/scrUser.h>

using namespace sci_cca;
#include <iostream>
using namespace std;

void
scrUser::setServices( const Services::pointer & srv )
{
  ComponentImpl::setServices( srv );

  if( !srv.isNull() )
    {
      array1<string> properties;
      properties.resize( 0 );

      cout << "Registering scrUserIn0 uses port\n";

      PortInfo::pointer pi = srv->createPortInfo( "scrUserIn0",
					 "sci_cca.scrInterface",
					 properties );
      srv->registerUsesPort( pi );
    }
}

void
scrUser::go()
{
  Port::pointer p = services_->getPort( "scrUserIn0" );

  scrInterface::pointer si = pidl_cast<scrInterface::pointer>( p );

  if( si.isNull() )
    {
      cout << "scrUser::go(): Error in get port\n";
      return;
    }

  int    cells = 1;
  double pressure = 2;
  double kgcat = 3;
  double NH3ratio = 4;
  double NH3 = 5;
  double flow = 6;
  double temp = 7;
  double NO = 8;
  double N2 = 9;
  double H2O = 10;
  double O2 = 11;
  double sum_of_all_others = 12;
  double heat_loss = 13;
  ::SSIDL::array1<double> kmol_s;
  double noreduction = 14;
  double new_temp = 15;
  double kmol_s_tot = 16;

  kmol_s.resize( 5 );
  kmol_s[0] = 1.0;
  kmol_s[1] = 2.0;
  kmol_s[2] = 3.0;
  kmol_s[3] = 4.0;
  kmol_s[4] = 5.0;

  si->exec( cells, pressure, kgcat, NH3ratio, NH3, flow, temp, NO,
	    N2, H2O, O2, sum_of_all_others, heat_loss, kmol_s,
	    noreduction, new_temp, kmol_s_tot );

  cout << "Returned values: " << flow << ", " << kmol_s_tot
       << ", " << kmol_s.size() << ", " << kmol_s[0] << "\n";

  services_->releasePort( "scrUserIn0" );
}

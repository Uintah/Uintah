
#include <testprograms/Component/framework/Builders/Builder.h>

#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/TestComponents/Sender.h>
#include <testprograms/Component/framework/TestComponents/Provider.h>
#include <testprograms/Component/framework/REI/scr.h>

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
Builder::menu()
{
  cout << "Builder options:\n";
  cout << "   1) Create component.\n";
  cout << "   2) Connect components.\n";
  cout << "   3) Read input script.\n"; 
  cout << "   l) List created components.\n"; 
  cout << "   q) Quit.\n"; 
  cout << "\n";
}

void
Builder::read_input_script()
{
  cout << "Not done yet.\n\n";
}

void
Builder::connect_components()
{
  cout << "Not done yet.\n\n";


  //    ConnectionServices connect_port = pidl_cast<ConnectionServices>(
  //                                          svc->getPort("ConnectionServices"));
//    if ( !connect_port ) {
//      cerr << "Could not get connection port\n";
//      return;
//    }

    //connect_port->connect( sid, "Uses", pid, "Provides" );
    
    //sender->go();
    
    //svc->releasePort( "ConnectionServices" );
}

void
Builder::create_component()
{
  char      command;
  bool      done;
  Component comp;
  string    name;

  do
    {
      done = true;

      cout << "Create what type of component:\n";
      cout << "  1) Provider, 2) Sender, 3) scr\n";
      cin >> command;
      cin.ignore( 100, '\n' ); 

      switch( command )
	{
	case '1':
	  comp = new Provider();
	  name = "Provider";
	  break;
	case '2':
	  comp = new Sender();
	  name = "Sender";
	  break;
	case '3':
	  name = "scr";
	  comp = new scr();
	  break;
	default:
	  done = false;
	}
    } while( !done );
  
  CCA::init( comp, name );
}

void
Builder::list_active_components()
{
  // Query the Registry for all active components...
  RegistryServices reg_port = pidl_cast<RegistryServices>(
                                       services_->getPort("RegistryServices"));
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
    services_->releasePort( "RegistryServices" );
    reg_port = 0;
  }
}

void
Builder::shutdown_framework()
{
  // Query the Registry for all active components...
  RegistryServices reg_port = pidl_cast<RegistryServices>(
                                       services_->getPort("RegistryServices"));
  if ( !reg_port ) {
    cerr << "Shutdown_Framework(): Could not get registry port\n";
    return;
  } else {

    reg_port->shutdown();

    // This is probably not connected after shutdown...
    //services_->releasePort( "RegistryServices" );
    reg_port = 0;
  }
}

void
Builder::ui()
{
  char command;

  while( 1 )
    {
      cout << "Command: "; cout.flush();
      cin >> command;
      cin.ignore( 100, '\n' );
      switch( command ) 
	{
	case '1' :
	  create_component();
	  break;
	case '2' :
	  connect_components();
	  break;
	case '3' :
	  read_input_script();
	  break;
	case 'l' :
	  list_active_components();
	  break;
	case 'q' :
	  shutdown_framework();
	  return;
	default:
	  cout << "\n";
	  cout << "Invalid option.\n\n";
	  menu();
	  break;
	}
    }
}

} // namespace sci_cca



#include <testprograms/Component/framework/Builders/Builder.h>

#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/TestComponents/Sender.h>
#include <testprograms/Component/framework/TestComponents/Provider.h>
#include <testprograms/Component/framework/REI/scr.h>
#include <testprograms/Component/framework/REI/scrUser.h>

#include <sstream>
#include <iostream>
using namespace std;

namespace sci_cca {

using std::cerr;

scrUser * scruser = 0;

Builder::Builder()
{
  // Hard code creation of a scrUser so that I can use it to call
  // the "go" function.  Go should actually be an interface...
  scruser = new scrUser();
  Component::pointer comp(scruser);
  CCA::init( comp, "scrUser" );
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
  array1<ComponentID::pointer> componentIds;

  RegistryServices::pointer reg_port = pidl_cast<RegistryServices::pointer>(
                                       services_->getPort("RegistryServices"));
  if ( reg_port.isNull() ) {
    cerr << "Could not get registry port\n";
    return;
  } else {

    cout << "\nComponents:\n"; 

    reg_port->getActiveComponentList( componentIds );

    cerr << componentIds.size() << " components returned:\n";

    for( unsigned int cnt = 0; cnt < componentIds.size(); cnt++ )
      {
	cerr << cnt << ": " << componentIds[ cnt ]->toString() << "\n";
      }
    services_->releasePort( "RegistryServices" );
    reg_port = 0;
  }

  unsigned int user, provider;

  cout << "\n";
  cout << "Enter uses component: "; cout.flush();
  cin >> user;

  cout << "Enter provides component: "; cout.flush();
  cin >> provider;

  // Unsigned int, thus no > 0 comparison.
  if( user >= componentIds.size() || provider >= componentIds.size() )
    {
      cout << "Invalid component.  Bye.\n";
      return;
    }

  BuilderServices::pointer build_port = 
     pidl_cast<BuilderServices::pointer>( services_->getPort("BuilderServices") );

  if ( build_port.isNull() ) {
    cerr << "Could not get builder port\n";
    return;
  }

  build_port->connect( componentIds[ user ], "scrUserIn0", 
		       componentIds[ provider ], "scrOut0" );
    
  services_->releasePort( "BuilderServices" );

  scruser->go();
}

void
Builder::create_component()
{
  char      command;
  bool      done;
  Component::pointer comp;
  string    name;

  do
    {
      done = true;

      cout << "Create what type of component:\n";
      cout << "  1) Provider, 2) Sender, 3) scr, 4) scrUser\n";
      cin >> command;
      cin.ignore( 100, '\n' ); 

      switch( command )
	{
	case '1':
	  comp = Component::pointer(new Provider());
	  name = "Provider";
	  break;
	case '2':
	  comp = Component::pointer(new Sender());
	  name = "Sender";
	  break;
	case '3':
	  name = "scr";
	  comp = Component::pointer(new scr());
	  break;
	case '4':
	  name = "scrUser";
	  comp = Component::pointer(new scrUser());
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
  RegistryServices::pointer reg_port = pidl_cast<RegistryServices::pointer>(
                                       services_->getPort("RegistryServices"));
  if ( reg_port.isNull() ) {
    cerr << "Could not get registry port\n";
    return;
  } else {
    array1<ComponentID::pointer> componentIds;
    reg_port->getActiveComponentList( componentIds );

    cerr << "Number of Components Returned: " << componentIds.size() << "\n";

    for( unsigned int cnt = 0; cnt < componentIds.size(); cnt++ )
      {
	cerr << cnt << ": " << componentIds[ cnt ]->toString() << "\n";
      }
    services_->releasePort( "RegistryServices" );
    reg_port = RegistryServices::pointer(0);
  }
}

void
Builder::shutdown_framework()
{
  // Query the Registry for all active components...
  RegistryServices::pointer reg_port = pidl_cast<RegistryServices::pointer>(
                                       services_->getPort("RegistryServices"));
  if ( reg_port.isNull() ) {
    cerr << "Shutdown_Framework(): Could not get registry port\n";
    return;
  } else {

    reg_port->shutdown();

    // This is probably not connected after shutdown...
    //services_->releasePort( "RegistryServices" );
    reg_port = RegistryServices::pointer(0);
  }
}

void
Builder::ui()
{
  char command;

  for(;;)
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


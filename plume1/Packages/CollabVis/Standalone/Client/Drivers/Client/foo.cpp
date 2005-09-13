#include <Network/NetInterface.h>
#include <Thread/Thread.h>
using namespace SemotusVisum;


int
main( int argc, char ** argv ) {
  
  cerr << NetInterface::getInstance().connectToServer( "localhost" ) << endl;
  cerr << "Connected!" << endl;
  sleep(2);
  NetInterface::getInstance().disconnectFromServer();
  cerr << "Disconnected!" << endl;
  sleep(10);
  Thread::exitAll( 1 );
}

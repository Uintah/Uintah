
#include <Packages/Uintah/Core/Tracker/TrackerServer.h>

#include <Core/Containers/StringUtil.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Thread.h>

#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

TrackerServer * TrackerServer::trackerServer_ = NULL;

void
TrackerServer::startTracking( unsigned int numClients )
{
  if( trackerServer_ != NULL ) {
    // ERROR, already intialized...
    cout << "ERROR: TrackerServer is already started... continuing, but something is probabl wrong...\n";
    return;
  }

  trackerServer_ = new TrackerServer( numClients ); 

  while( !trackerServer_->shutdown_ ) {
    string result;
    int size = trackerServer_->sockets_[0]->read( result );

    if( size <= 0 ) {
      break;
    }

    cout << "from: " << trackerServer_->sockets_[0]->getSocketInfo() << "\n";
    cout << "just received: " << result;
  }

}

TrackerServer::TrackerServer( unsigned int numClients ) :
  shutdown_( false )
{
  Socket socket;
  socket.create();
  cout << "created\n";
  socket.bind( TRACKER_PORT );
  cout << "bound\n";
  socket.listen();
  cout << "listened\n";

  for( unsigned int pos = 0; pos < numClients; pos++ ) {
    sockets_.push_back( new Socket() );
    cout << "accepting\n";
    socket.accept(  *sockets_[pos] );
    cout << "accepted\n";
  }
}

TrackerServer::~TrackerServer()
{
  // Clean up.
  for( unsigned int pos = 0; pos < sockets_.size(); pos++ ) {
    delete sockets_[ pos ];
  }
}

void
TrackerServer::quit() 
{
  trackerServer_->shutdown_ = true;
}

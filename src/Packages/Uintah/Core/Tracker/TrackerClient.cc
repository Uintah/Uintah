
#include <Packages/Uintah/Core/Tracker/TrackerClient.h>

#include <iostream>
#include <sstream>

using namespace std;
using namespace Uintah;

TrackerClient * TrackerClient::trackerClient_ = NULL;

bool
TrackerClient::initialize( const std::string & host )
{
  if( trackerClient_ != NULL ) {
    // ERROR, already intialized...
    cout << "ERROR: TrackerClient is already initialized... continuing, but something is probabl wrong...\n";
    return true;
  }
  trackerClient_ = new TrackerClient();

  trackerClient_->socket_.create();

  cout << "Attempting to connect to " << host << " on port " << TRACKER_PORT << "\n";
  bool result = trackerClient_->socket_.connect( host, TRACKER_PORT );
  cout << "connect call finished.\n";
  if( !result ) {
    // FIXME
    cout << "socket connect failed\n";
    trackerClient_ = NULL;
    return false;
  }
  return true;
}

TrackerClient::TrackerClient() :
  sendLock_( "TrackerClient send lock" )
{
}

void
TrackerClient::trackMPIEvent( MPIMessageType mt, std::string & variable, std::string & info )
{
  TrackerClient * tc = trackerClient_;

  if( !tc ) return;

  tc->sendLock_.lock();  

  map<string,short>::iterator iter = tc->variableToIndex_.find( variable );

  short index;

  if( iter == tc->variableToIndex_.end() ) {
    // Must send the full variable name the first time.
    index = tc->variableToIndex_.size();
    tc->variableToIndex_[ variable ] = index;
  }
  else {
    index = iter->second;
  }

  tc->sendLock_.unlock();

} // end trackMPIEvent()

void
TrackerClient::trackEvent( GeneralMessageType mt, int value )
{
  TrackerClient * tc = trackerClient_;

  if( !tc ) return;

  //tc->sendLock_.lock();  
  //tc->sendLock_.unlock();  

  cout << "not implemented yet\n";
}

void
TrackerClient::trackEvent( GeneralMessageType mt, double value )
{
  TrackerClient * tc = trackerClient_;

  if( !tc ) return;

  ostringstream msg;
  msg << Tracker::toString( mt ) << " " << value << "\n";

  tc->sendLock_.lock();  

  tc->socket_.write( msg.str() );

  tc->sendLock_.unlock();  
}

void
TrackerClient::trackEvent( GeneralMessageType mt, const string & value )
{
  TrackerClient * tc = trackerClient_;

  if( !tc ) return;

  //tc->sendLock_.lock();  
  //tc->sendLock_.unlock();  

  cout << "not implemented yet\n";
}

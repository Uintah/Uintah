/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Tracker/TrackerClient.h>

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

/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <Core/Tracker/TrackerServer.h>

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

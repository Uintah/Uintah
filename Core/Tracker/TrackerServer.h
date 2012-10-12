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


#ifndef UINTAH_CORE_TRACKER_TRACKERSERVER_H
#define UINTAH_CORE_TRACKER_TRACKERSERVER_H

#include <Core/Tracker/Tracker.h>

#include <Core/Util/Socket.h>

#include <string>
#include <vector>

namespace Uintah {

/**************************************
     
CLASS
  TrackerServer - Singleton Class
 
  Used as a separate tracking process that sus instantiations can talk to 
  to report their progress.
 
GENERAL INFORMATION
 
  TrackerServer.h
 
  J. Davison de St. Germain
  SCI Institute
  University of Utah
 
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
 
 
KEYWORDS
  Progress Tracking
 
DESCRIPTION
  Long description...
 
WARNING
 
****************************************/

class TrackerServer : public Tracker{

public:

  // Tells the TrackerServer (singleton class) to open up a socket and
  // wait for clients to connect... Then listen for their updates.  A
  // separate thread (NOT IMPLEMENTED YET!) is created that will do this.
  // 'numClients' is the number of clients that the Server should expect 
  // to hear from.

  static void startTracking( unsigned int numClients );

  // Shuts down the server.
  static void quit();

private:

  TrackerServer( unsigned int numClients );
  ~TrackerServer();

  static TrackerServer * trackerServer_;

  std::vector< SCIRun::Socket * > sockets_;

  bool shutdown_;

}; // end class TrackerServer
  
} // end namespace Uintah

#endif


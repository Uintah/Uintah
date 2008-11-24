
#ifndef UINTAH_CORE_TRACKER_TRACKERSERVER_H
#define UINTAH_CORE_TRACKER_TRACKERSERVER_H

#include <Packages/Uintah/Core/Tracker/Tracker.h>

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
 
  Copyright (C) 2008 SCI Group
 
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


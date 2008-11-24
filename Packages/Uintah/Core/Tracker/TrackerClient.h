
#ifndef UINTAH_CORE_TRACKER_TRACKERCLIENT_H
#define UINTAH_CORE_TRACKER_TRACKERCLIENT_H

#include <string>
#include <map>

#include <Core/Thread/Mutex.h>
#include <Core/Util/Socket.h>

#include <Packages/Uintah/Core/Tracker/Tracker.h>

namespace Uintah {

/**************************************
     
CLASS
  TrackerClient
 
  Allows for easily setting up communication to a TrackerServer
 
GENERAL INFORMATION
 
  TrackerClient.h
 
  J. Davison de St. Germain
  SCI Institute
  University of Utah
 
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
 
  Copyright (C) 2008 SCI Group
 
KEYWORDS
  Progress Tracking
 
DESCRIPTION
  A singleton class.
 
WARNING
 
****************************************/

class TrackerClient : public Tracker {

public:

  // Tells the TrackerClient (singleton class) to open up its socket
  // to the TrackerServer on 'server' (can be ip addr or host name).  
  // 
  // Initialize() MUST BE CALLED BEFORE any other functions.

  static bool initialize( const std::string & server );

  static void trackMPIEvent( MPIMessageType mt, std::string & variable, std::string & info );

  static void trackEvent( GeneralMessageType mt, int value );
  static void trackEvent( GeneralMessageType mt, double value );
  static void trackEvent( GeneralMessageType mt, const std::string & value );

private:

  // This is a singleton class... don't construct it.
  TrackerClient();

  // Keeps track of a string (variable name) to short integer (unique index) map so that
  // messages can be shorter (than sending name every time).

  std::map< std::string, short > variableToIndex_;

  SCIRun::Mutex  sendLock_;

  SCIRun::Socket socket_;

  static TrackerClient * trackerClient_;

}; // end class TrackerClient
  
} // end namespace Uintah

#endif


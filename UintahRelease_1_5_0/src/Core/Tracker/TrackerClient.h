/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_CORE_TRACKER_TRACKERCLIENT_H
#define UINTAH_CORE_TRACKER_TRACKERCLIENT_H

#include <string>
#include <map>

#include <Core/Thread/Mutex.h>
#include <Core/Util/Socket.h>

#include <Core/Tracker/Tracker.h>

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


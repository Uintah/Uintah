
#ifndef UINTAH_CORE_TRACKER_TRACKER_H
#define UINTAH_CORE_TRACKER_TRACKER_H

#include <string>

namespace Uintah {

/**************************************
     
CLASS
  Tracker
 
  Common stuff for Tracker Clients and Servers
 
GENERAL INFORMATION
 
  Tracker.h
 
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
   
class Tracker {

public:

  enum MPIMessageType { MPI_BCAST, MPI_SEND, MPI_RECV };

  //  VARIABLE_NAME    - Sends a variable's string name to the server so that it will be linked with a specific index.
  //  TIMESTEP_STARTED - Timestep just started.

  enum GeneralMessageType { VARIABLE_NAME, TIMESTEP_STARTED };

  static std::string toString( MPIMessageType mt );
  static std::string toString( GeneralMessageType mt );

protected:

private:

}; // end class Tracker

} // end namespace Uintah

#define TRACKER_PORT 5555
  

#endif


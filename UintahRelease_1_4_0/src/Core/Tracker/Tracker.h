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


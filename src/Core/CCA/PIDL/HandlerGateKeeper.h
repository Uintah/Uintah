/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  HandlerGateKeeper.h: Class which allows only one component invocation at a time
 *                       in the case of parallel components
 *                       .
 *                   
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef CCA_PIDL_HandlerGateKeeper_h
#define CCA_PIDL_HandlerGateKeeper_h

#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <string>
#include <map>
#include <vector>

namespace SCIRun {
/**************************************
 
CLASS
   HandlerGateKeeper
   
KEYWORDS
   PIDL, Server
   
DESCRIPTION
   The HandlerGateKeeper class is associated with a server
   and it's purpose is to make collective component invocations
   be serviced one-by-one. The parallel component's processes
   can make several invocations to a callee parallel components.
   This class differentiates between calls of one parallel component
   and another.
****************************************/
  class HandlerGateKeeper {

  public:
    
    /////////
    // Default constructor
    HandlerGateKeeper();

    //////////
    // Destructor which clears all current storage entries
    ~HandlerGateKeeper(); 

    ///////
    // Gets access to critical section for all processes invoking with the
    // same session ID
    int getTickets(int handler_num,::std::string sessionID, int number_of_calls);

    ///////
    // Releases access to critical section after called number_of_calls for 
    // all processes invoking with the same session ID.
    void releaseOneTicket(int handler_num);

  private:

    //String comparison function for std::map
    struct ltint
    {
      bool operator()(const int i1, const int i2) const
      {
        return (i1 < i2);
      }
    };
    
    //////////////
    //A map of handler numbers to current session ID of the component 
    //that has access to that method. 
    typedef ::std::map<int, ::std::string, ltint> IDMap;
    typedef ::std::map<int, ::std::string>::value_type IDMap_valType;
    IDMap currentID;

    //////////
    // Number of calls associated with a specific method/handler
    ::std::vector< int> callsToGo;

    /////////
    // Semaphore used to hold out of order invocations
    Semaphore d_gate_sema;

    ///////
    // Mutex used to control access of the callsToGo variable above
    Mutex d_gate_mutex;

  };
} // End namespace SCIRun

#endif





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
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include "HandlerGateKeeper.h"
#include <iostream>
using namespace SCIRun;

HandlerGateKeeper::HandlerGateKeeper()
  :currentID(64,' '), callsToGo(0), d_gate_sema("HandlerGateKeeper Semaphore",0),
   d_gate_mutex("HandlerGateKeeper Mutex")
{
}

HandlerGateKeeper::~HandlerGateKeeper()
{
}

int HandlerGateKeeper::getTickets(::std::string sessionID, int number_of_calls)
{
  d_gate_mutex.lock();
  if((callsToGo > 0)&&(currentID != sessionID)) {
    d_gate_mutex.unlock();
    d_gate_sema.down(); 	
  }
  if(currentID != sessionID) {
    currentID = sessionID;
    ::std::cout << "current sessionID is '" << currentID << "', numCalls=" << number_of_calls << "\n";
    callsToGo = number_of_calls;
  }
  d_gate_mutex.unlock();
  return 0;
}

void HandlerGateKeeper::releaseOneTicket() 
{
  callsToGo--;
  ::std::cout << callsToGo << " calls to go\n";
  if(callsToGo == 0) {
    currentID.clear();
    d_gate_sema.up();	  
  }
}


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
  :d_gate_sema("HandlerGateKeeper Semaphore",0),
   d_gate_mutex("HandlerGateKeeper Mutex")
{
}

HandlerGateKeeper::~HandlerGateKeeper()
{
}

int HandlerGateKeeper::getTickets(int handler_num, ::std::string sessionID, int number_of_calls)
{
  if(callsToGo.size() <= handler_num) callsToGo.resize(handler_num+1);
  d_gate_mutex.lock();
  if((callsToGo[handler_num] > 0)&&(currentID.find(handler_num) != currentID.end())
		                 &&(currentID[handler_num] != sessionID)) {
    d_gate_mutex.unlock();
    //::std::cerr << "I shot the sheriff, currentID =" << currentID[handler_num] << ", sessionID=" << sessionID << "\n";
    d_gate_sema.down(); 	
    d_gate_mutex.lock();
  }
  if((currentID.find(handler_num) == currentID.end())||(currentID[handler_num] != sessionID)) {
    currentID.insert(IDMap_valType(handler_num, sessionID));
    if(number_of_calls == 0) number_of_calls++; 
    //::std::cout << "current sessionID for handler " << handler_num << "is '" 
    //		<< currentID[handler_num] << "', numCalls=" << number_of_calls << "\n";
    callsToGo[handler_num] = number_of_calls;
  }
  d_gate_mutex.unlock();
  return 0;
}

void HandlerGateKeeper::releaseOneTicket(int handler_num) 
{
  d_gate_mutex.lock();
  //Decrement number of calls to go
  callsToGo[handler_num]--;
  d_gate_mutex.unlock();

  //::std::cout << callsToGo[handler_num] << " calls to go\n";
  if(callsToGo[handler_num] == 0) {
    d_gate_mutex.lock();
    //reset ID
    currentID[handler_num].clear();
    d_gate_mutex.unlock();
    //Lift sema in getTickets()
    d_gate_sema.up();	  
  }
}


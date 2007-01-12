/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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


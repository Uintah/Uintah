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
#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>

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





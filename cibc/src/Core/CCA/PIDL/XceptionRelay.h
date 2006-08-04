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
 *  XceptionRelay.h: Class which propagates imprecise exceptions
 *                       
 *                       .
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef CCA_PIDL_XceptionRelay_h
#define CCA_PIDL_XceptionRelay_h

#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/Comm/Intra/IntraComm.h>
#include <Core/CCA/Comm/Message.h>
#include <map>

#define MAX_X_MSG_LENGTH 300

namespace SCIRun {
/**************************************
 
CLASS
   XceptionRelay
   
KEYWORDS
   PIDL, Proxy, Exception
   
DESCRIPTION
   The XceptionRelay class is associated with a proxy
   and it's purpose is to provide imprecise exception 
   semantics via asynchronous messaging. 

****************************************/
  class XceptionRelay {

  public:
    
    /////////
    // Constructor
    XceptionRelay(ProxyBase* pb);

    //////////
    // Destructor which clears all current storage entries
    ~XceptionRelay(); 

    ///////
    // Broadcast an exception to all cohorts. Before throwing
    // the exception we run a synchronization (tournament 
    // barrier) to make sure nobody else had an exception that
    // superseeds this one. If there is a more important 
    // exception, we will try to retreive it.
    // This method does not throw the exception, this is done in SIDL.
    void relayException(int* x_id, Message** message);

    ///////
    // Performs a check whether exceptions exist to be thrown
    // at the current lineID.
    // Ignores all exceptions that are to come later than the 
    // current lineID. When an exception is about to be thrown, 
    // a level of synchronization and comparisson is done to 
    // make sure the right exception is being thrown.
    int checkException(Message** _xMsg);

    
    ///////
    // Checks whether a message for an exception has arrived. 
    int readException(Message** _xMsg, int* _xlineID);

    /////
    // Retreives the current lineID 
    int getlineID();


    ////
    // Resets internal lineID counter
    void resetlineID();
    
  private:

    ////////
    // Representation for each exception
    struct Xception {
      //int lineID;
      int xID;
      Message* xMsg;
      bool thrown;
    };
    
    //int comparison function for std::map
    struct ltint
    {
      bool operator()(const int i1, const int i2) const
      {
        return (i1 < i2);
      }
    };
    
    //////////////
    //
    typedef ::std::map<int, Xception, ltint> XDB;
    typedef ::std::map<int, Xception>::value_type XDB_valType;
    XDB xdb;

    /////////
    // A counter which is incremented by checkException(). Since
    // 
    int lineID;

    ///////
    // A pointer to the intracommunicator used to provide communication
    // primitives between the nodes
    ProxyBase* mypb;

    char sbuf[MAX_X_MSG_LENGTH];
    char rbuf[MAX_X_MSG_LENGTH];
  };
} // End namespace SCIRun

#endif





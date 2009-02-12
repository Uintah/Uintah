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
 *  MxNArrSynch.h 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   August 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef CCA_PIDL_MxNArrSynch_h
#define CCA_PIDL_MxNArrSynch_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MxNScheduleEntry.h>
#include <map>

/**************************************
				       
  CLASS
    MxNArrSynch
   
  DESCRIPTION
    MxNArrSynch is concerned with providing
    synchronization for MxN data transfer
    and collective method invocation. 

****************************************/

namespace SCIRun {  

  class MxNScheduleEntry;

  class MxNArrSynch {
  public:

    //////////////
    // Default constructor accepts a pointer to the MxNScheduleEntry
    // class encapsulating the distribution schedule.
    MxNArrSynch(MxNScheduleEntry* sched, MxNArrayRep *myrep);

    /////////////
    // Destructor which takes care of various memory freeing
    virtual ~MxNArrSynch();
    
    //////////
    // Retrieves pointer to the actual array without regard to the
    // state of that array.
    void* getArray();

    /////////
    // It sets a new array pointer (for in arguments) to an array 
    // of the appropriate size and dimension. [CALLEE ONLY]
    void setNewArray(void** a_ptr);

    /////////
    // It sets a pointer (for out arguments) to the recieved array
    // [CALLEE ONLY]
    void setArray(void** a_ptr);

    //////////
    // Retrieves pointer to the actual array only if the array pointer
    // does not equal NULL. [CALLEE ONLY] 
    void* getArrayWait();
    
    /////////
    // Blocks until data redistribution is complete and placed
    // within the array. After that is complete we return the
    // pointer to the array. [CALLEE ONLY]
    void* waitCompleteInArray(MxNArrayRep *myrep);

    /////////
    // Blocks until data redistribution are sent to the callers
    // [CALLEE ONLY]
    void waitCompleteOutArray(MxNArrayRep *myrep);

    /////////
    // This method is called when we have recieved the distribution
    // from a particular object denoted by its rank. It increses the 
    // recieve count
    // [CALLEE ONLY]
    void doReceive(int rank);

    /////////
    // This method is called when we have sent the distribution
    // to a particular object denoted by its rank. It increses the 
    // send count
    // [CALLEE ONLY]
    void doSend(int rank);

    ///////////
    // Prints the contents of this object
    void print(std::ostream& dbg);

  private:

    ////////
    // Reference to MxNScheduleEntry, needed to access distribution schedule
    MxNScheduleEntry* _sched;

    ////////
    // A void pointer to the actual array. Used only if this object
    // is associated with a callee. The main purpose of this is to
    // provide a main storage for the recieved array so that copying 
    // will be prevented. This pointer is set and retrieved only by
    // the sidl generated code.
    void* arr_ptr;

    //////////
    // The condition variable is used to block until all data has been recieved.
    ConditionVariable recvCondition;

    //////////
    // The condition variable is used to block until all data has been sent.
    ConditionVariable sendCondition;

    //////////
    // The condition variable is used to block until the arr_ptr is set
    ConditionVariable arrCondition;

    ////////
    // Mutex used to control access of the array pointer
    Mutex arr_mutex;

    //////////
    // expected count of data redistributions that should be received/sent
    int expected_count;

    ///////
    // Mutex used to control access of the receive flags on the 
    // array representations
    Mutex recv_mutex;

    ///////
    // count of received distribution
    int recv_count;

    ///////
    // Mutex used to control access of the receive flags on the 
    // array representations
    Mutex send_mutex;

    ///////
    // count of sent distribution
    int send_count;

    ///////
    // Used to determine whether we should let the array pointer to be set
    bool allowArraySet;

  };
} // End namespace SCIRun

#endif








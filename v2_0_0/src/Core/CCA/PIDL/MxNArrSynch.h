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
#include <Core/Thread/Mutex.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MxNScheduleEntry.h>

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
    MxNArrSynch(MxNScheduleEntry* sched);

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
    void* waitCompleteArray();

    /////////
    // This method is called when we have recieved the distribution
    // from a particular object denoted by its rank. It sets the 
    // recieve flag on that distribution in the caller's descriptorList.
    // [CALLEE ONLY]
    void doReceive(int rank);

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
    // The semaphore used to block until all data has been recieved.
    // After all data is received the call to getCompleteArray() will
    // go through.
    Semaphore recv_sema;

    //////////
    // The semaphore used to block until setArray() has been called
    // in the case of getArrayWait()
    Semaphore arr_wait_sema;

    ////////
    // Mutex used to control access of the array pointer
    Mutex arr_mutex;

    ///////
    // Mutex used to control access of the receive flags on the 
    // array representations
    Mutex recv_mutex;

    ///////
    // Used to determine whether we should let the array pointer to be set
    bool allowArraySet;

  };
} // End namespace SCIRun

#endif








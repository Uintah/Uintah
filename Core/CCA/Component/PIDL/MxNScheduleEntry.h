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
 *  MxNScheduleEntry.h 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002 
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef Component_PIDL_MxNScheduleEntry_h
#define Component_PIDL_MxNScheduleEntry_h

#include <string>
#include <vector>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Core/CCA/Component/PIDL/MxNArrayRep.h>

#define DEBUG_THE_SEMAS

/**************************************
				       
  CLASS
    MxNScheduleEntry
   
  DESCRIPTION
    This class encapsulates information associated
    with a particular array distribution within a
    given object. The class does not differentiate
    between a caller and callee object. It also does
    not calculate a schedule for the given distribution.
    The class is mainly used to contain all pertinent 
    information about the distribution and also to
    provide some bookkeeping and synchronization.
    Most of this class' methods are used withing the 
    MxNScheduler and NOT within the sidl code.

****************************************/

namespace SCIRun {  
  
  ////////////
  // Enumerated type denoting caller and callee. Used to
  // tag this object as caller or callee in respect
  // to the distribution.
  typedef enum { caller, callee } sched_t;

  ///////////
  // List of array descriptions
  typedef std::vector<MxNArrayRep*> descriptorList;

  class MxNScheduleEntry {
  public:

    //////////////
    // Constructor accepts the name of the distribution
    // and the purpose of this object in the form caller/callee 
    MxNScheduleEntry(std::string n, sched_t st);

    /////////////
    // Destructor which takes care of various memory freeing
    virtual ~MxNScheduleEntry();
    
    ////////////
    // Returns true if this object is associated with a caller/callee
    bool isCaller();
    bool isCallee();

    ////////////
    // Adds a caller/callee array description to the list
    void addCallerRep(MxNArrayRep* arr_rep);
    void addCalleeRep(MxNArrayRep* arr_rep);
    
    ///////////
    // Retrieves an caller/callee array description by a given 
    // index number
    MxNArrayRep* getCallerRep(unsigned int index);
    MxNArrayRep* getCalleeRep(unsigned int index);

    //////////
    // Method is called when a remote Array description is recieved.
    // The number of remote objects (size) is passed in to check
    // whether that many descriptions have already been recieved.
    // If that is the case, we raise a synchronization semaphore.
    void reportMetaRecvDone(int size);

    //////////
    // Retrieves pointer to the actual array without regard to the
    // state of that array.
    void* getArray();

    /////////
    // Calculate the redistribution schedule given the current array
    // representation this class has (if it had not been calculated
    // before). Returns a the schedule. [CALLER ONLY]
    descriptorList makeSchedule(); 
    
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
    /////////
    // Name of the distribution this object is associated with
    std::string name;

    /////////
    // Role (caller/callee) that this object helps fulfil
    sched_t scht;

    /////////
    // List of caller/callee array representations. Used to store
    // both the local and remote representations
    descriptorList caller_rep;
    descriptorList callee_rep;

    /////////
    // The computed distribution schedule.
    descriptorList sched;

    /////////
    // Used to determine whether or not the schedule has been calculated
    bool madeSched;

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

    /////////
    // Semaphore used to separate the metadata (remote array descriptions)
    // reception stage from the actual redistribution.
    Semaphore meta_sema;

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

#ifdef DEBUG_THE_SEMAS
    int recv_sema_count;
    int arr_wait_sema_count;
    int meta_sema_count;
    void printSemaCounts();
#endif

  };
} // End namespace SCIRun

#endif








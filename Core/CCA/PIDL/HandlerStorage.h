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
 *  HandlerStorage.h: Class which preserves data in between separate
 *                    invocations of the same EP handler.
 *                   
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   Sept 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef CCA_PIDL_HandlerStorage_h
#define CCA_PIDL_HandlerStorage_h

#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
/**************************************
 
CLASS
   HandlerStorage
   
KEYWORDS
   PIDL, Server
   
DESCRIPTION
   The HandlerStorage class is associated with a server
   and it's purpose is to store pointers to data in a location
   referenced by the handler number. The classes' purpose at
   this point is to store return data for NOCALLRET type of calls.
   The reason this classes was added is that current version of Globus
   Nexus used in the system does not allow for any kind of buffer copying.
   When the underlying communication mechanism is changed we can do away
   with this call and implement a more effecient buffer copy instead.
****************************************/
  class HandlerStorage {

  public:
    
    //String comparison function for std::map
    struct ltint
    {
      bool operator()(const int i1, const int i2) const
      {
	return (i1 < i2);
      }
    };

    //////////////
    //A map of scheduleEntries to distibution names
    typedef std::vector< void*> voidvec;
    typedef std::map<int, voidvec, ltint> dataList;
    
    HandlerStorage();

    //////////
    // Destructor which clears all current storage entries
    ~HandlerStorage(); 

    /////////
    // Clears storage entries. Call without a handler number
    // clears all of the current entries.
    void clear(int handler_num = 0);

    ////////
    // Adds data to the storage queue 
    void add(int handler_num, int queue_num, void* data);

    ///////
    // Retrieves queue element from storage
    void* get(int handler_num, int queue_num);

  private:

    //////////
    // Used to preserve some data in this 
    // object in order to save message between separate
    // calls.
    dataList d_data;

    /////////
    // Semaphore used to wait for an insert
    SCIRun::Semaphore d_data_sema;

    ////////
    // Mutex used to control access of the d_data variable abover
    Mutex d_data_mutex;

    ///////
    // Count the number of threads that are waiting for the semaphore.
    int threadcount;
  };
} // End namespace SCIRun

#endif





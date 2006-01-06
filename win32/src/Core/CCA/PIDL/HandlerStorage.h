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
#include <string>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/CCA/PIDL/ProxyBase.h>
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
    struct ltstr
    {
      bool operator()(const std::string s1, const std::string s2) const
      {
	return (s1.compare(s2) < 0);
      }
    };

    //////////////
    //A map of scheduleEntries to distibution names
    typedef std::vector< void*> voidvec;
    typedef std::map<std::string, voidvec, ltstr> dataList;
    
    HandlerStorage();

    //////////
    // Destructor which clears all current storage entries
    ~HandlerStorage(); 

    /////////
    // Clears storage entries. Call without a handler number
    // clears all of the current entries.
    void clear();

    ////////
    // Adds data to the storage queue 
    //    void add(int handler_num, int queue_num, void* data, std::string uuid, int callID, int numCalls);
    void add(int handler_num, int queue_num, void* data, ProxyID uuid, int callID, int numCalls);

    ///////
    // Retrieves queue element from storage
    //    void* get(int handler_num, int queue_num, std::string uuid, int callID);
    void* get(int handler_num, int queue_num, ProxyID uuid, int callID);

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





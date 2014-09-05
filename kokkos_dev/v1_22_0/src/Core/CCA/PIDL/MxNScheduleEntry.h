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

#ifndef CCA_PIDL_MxNScheduleEntry_h
#define CCA_PIDL_MxNScheduleEntry_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MxNArrSynch.h>

/**************************************
				       
  CLASS
    MxNScheduleEntry
   
  DESCRIPTION
    This class encapsulates information associated
    with a particular array distribution within a
    given object. The class does differentiate
    between a caller and callee object. It also 
    calculates a schedule for the given distribution.
    The class is mainly used to contain all pertinent 
    information about the distribution and also to
    provide some bookkeeping.
    Most of this class' methods are used withing the 
    MxNScheduler and NOT within the sidl code.

****************************************/

namespace SCIRun {  

  class MxNArrSynch;

  //String comparison function for std::map
  struct ltstr
  {
    bool operator()(const std::string s1, const std::string s2) const
    {
      return (s1.compare(s2) < 0);
    }
  };
  
  ///////////
  // List of array descriptions
  typedef std::vector<MxNArrayRep*> descriptorList;
  
  //////////
  // List of array synchs
  typedef std::map<std::string, MxNArrSynch*, ltstr> synchList;

  class MxNScheduleEntry {
  public:

    //ArrSynch needs access to local data in order to synchronize
    friend class MxNArrSynch;

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

    /////////
    // Calculate the redistribution schedule given the current array
    // representation this class has (if it had not been calculated
    // before). Returns a the schedule. [CALLER ONLY]
    descriptorList* makeSchedule(); 
    
    //////////
    // Method is called when a remote Array description is recieved.
    // The number of remote objects (size) is passed in to check
    // whether that many descriptions have already been recieved.
    // If that is the case, we raise a synchronization semaphore.
    void reportMetaRecvDone(int size);

    //////////
    // Clears existing arrays of MxNArrayReps
    void clear(sched_t sch); 

    ///////////
    // Prints the contents of this object
    void print(std::ostream& dbg);

    //////////
    // List of synchronization object corresponding uniquely to an
    // 'callid' invocation by component with 'uuid'.
    synchList s_list;

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

    /////////
    // Semaphore used to separate the metadata (remote array descriptions)
    // reception stage from the actual redistribution.
    Semaphore meta_sema;
  };
} // End namespace SCIRun

#endif








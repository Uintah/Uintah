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
 *   Kostadin Damevski & Keming Zhang
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
#include <Core/CCA/PIDL/MxNMetaSynch.h>

/**************************************
				       
  CLASS
    MxNScheduleEntry
   
  DESCRIPTION
    This class encapsulates information associated
    with a particular array distribution within a
    given object. The class is mainly used to contain 
    all pertinent information about the distribution 
    and also to provide some bookkeeping.
    Most of this class' methods are used withing the 
    MxNScheduler and NOT within the sidl code.

****************************************/

namespace SCIRun {  

  class MxNArrSynch;
  class MxNMetaSynch;

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
  // the key is call ID.
  typedef std::map<int, MxNArrSynch*> synchList;

  class MxNScheduleEntry {
  public:

    //ArrSynch and MetaSynch need access to local data in order to synchronize
    friend class MxNArrSynch;
    friend class MxNMetaSynch;

    //////////////
    // Default Constructor 
    MxNScheduleEntry();

    /////////////
    // Destructor which takes care of various memory freeing
    virtual ~MxNScheduleEntry();
    
    ////////////
    // Adds a caller/callee array description to the list
    void addRep(MxNArrayRep* arr_rep);
    
    ///////////
    // Retrieves an caller/callee array description by a given 
    // index number
    MxNArrayRep* getRep(unsigned int index);


    /////////
    // Calculate the redistribution schedule given the current array
    // representation this class has (if it had not been calculated
    // before). Returns the schedule.
    descriptorList* makeSchedule(MxNArrayRep* this_rep); 

    //////////
    // Clears existing arrays of MxNArrayReps
    void clear();

    ///////////
    // Prints the contents of this object
    void print(std::ostream& dbg);

    //////////
    // List of synchronization object corresponding uniquely to an
    // 'callid' invocation by component with 'uuid'.
    // used for server scheduler only
    synchList s_list;

    //////////
    // synchronization object used for meta data
    // used for server scheduler only
    MxNMetaSynch* meta_sync;

  private:
    // List of caller/callee array representations. 
    // Used to store remote representations
    descriptorList rep;

    /////////
    // The computed distribution schedule.
    descriptorList sched;

    /////////
    // Used to determine whether or not the schedule has been calculated
    bool madeSched;

  };
} // End namespace SCIRun

#endif








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
 *  MxNMetaSynch.h 
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   August 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef CCA_PIDL_MxNMetaSynch_h
#define CCA_PIDL_MxNMetaSynch_h

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
    MxNMetaSynch
   
  DESCRIPTION
    MxNMetaSynch is concerned with providing
    synchronization for MxN meta data transfer, 
    i.e. setCallerDistribution.
****************************************/

namespace SCIRun {  

  class MxNScheduleEntry;

  class MxNMetaSynch {
  public:

    //////////////
    // Default constructor accepts the size of proxy
    MxNMetaSynch(int size);

    /////////////
    // Destructor which takes care of various memory freeing
    virtual ~MxNMetaSynch();
    
    /////////////
    // When a setCallerDistribution handler is called, enter()
    // should be called.
    void enter();

    /////////
    // When a setCallerDistribution handler is about to finish,
    // leave() should be called.
    void leave(int rank);

    /////////
    // Blocks until setCallerDistribution is completely done
    void waitForCompletion();
  private:

    //////////
    // indicates if setCallerDistribution is completed.
    bool completed;

    //////////
    // number callers still inside the setCallerDistribution
    int numInside;

    //////////
    // array representations
    Mutex count_mutex;

    ConditionVariable metaCondition;

    ///////
    // Flags indicating distribution from which rank is done.
    // flag=0 initially, and if flag>1, that means two or more 
    // consecutive setCallerDistribution() are called. It is ok 
    // to let the later ones to overwrite the ealier ones. 
    // And when all flags are the same and larger than 0, 
    // some setCallerDistribution is done. 
    std::vector<int> recv_flags;
  };
} // End namespace SCIRun

#endif








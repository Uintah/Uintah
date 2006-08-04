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
 *  MxNScheduler.h 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002 
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef CCA_PIDL_MxNScheduler_h
#define CCA_PIDL_MxNScheduler_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <fstream>
#include <sgi_stl_warnings_on.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MxNArrSynch.h>
#include <Core/CCA/PIDL/MxNMetaSynch.h>
#include <Core/CCA/PIDL/MxNScheduleEntry.h>

/**************************************
				       
  CLASS
    MxNScheduler
   
  DESCRIPTION
    An instance of this class is associated with each component object
    that participates in an MxN redistribution regardless wether the
    object has a callee or caller role. The instantion is managed 
    through the sidl generated code. The class provides both caller
    and callee specific methods and it can be used for more than one
    distribution. 

****************************************/

namespace SCIRun {  
  class ProxyID;


  ////////////////////
  // R = process rank; S = process size; L = array length in this dimension
  #define BLOCK(R,S,L) MxNScheduler::makeBlock(R,S,L)
  #define CYCLIC(R,S,L) MxNScheduler::makeCyclic(R,S,L)

  //////////////
  //A map of scheduleEntries to distibution names
  //The string has different meaning for server scheduler and client scheduler:
  //It is the distribution name + proxy_uuid for the server's caller scheduler
  //entries, and only distribution name for anything else. However, to make the
  //the program easier to understand, callers' schedule entries always stored
  //distribution name+proxy_uuid, and callees' with only distribution name.
  typedef std::map<std::string, MxNScheduleEntry*, ltstr> schedList;
  typedef std::map<std::string, MxNArrayRep* , ltstr> arrRepList;

  class MxNScheduler {    
  public:
    
    MxNScheduler(sched_t sch=caller);
    virtual ~MxNScheduler();

    ////////////
    // Accepts an array representation describing a callee/callee
    // and assigns it to the appropriate ScheduleEntry for that 
    // distribution name.
    void setCalleeRepresentation(std::string distname, 
				 MxNArrayRep* arrrep);

    ////////////
    // Accepts an array representation describing a callee/callee
    // and assigns it to the appropriate ScheduleEntry for that 
    // distribution name + proxy_uuid
    void setCallerRepresentation(std::string distname, ProxyID uuid, MxNArrayRep* arrrep);
    
    ////////////
    // Checks wether we are playing a caller/callee method
    // from the perspective of the distribution and 
    // gets the native distribution representation
    MxNArrayRep* calleeGetCalleeRep(std::string distname);
    MxNArrayRep* callerGetCallerRep(std::string distname, ProxyID uuid);

    /////////
    // Static methods that retrieve a appropriate Index
    // to each process participating in a defined distribution
    // such as block, cyclic, etc.
    static Index* makeBlock(int rank, int size, int length);
    static Index* makeCyclic(int rank, int size, int length);
    

    ////////////
    // (Callee Metods)
    // These methods create a posibility to assign and access
    // an redistibuted array. This array is used throughout all
    // of the separate data receptions in order to eliminate 
    // copying
    // SeeAlso: similar methods in MxNScheduleEntry Class
    void setArray(std::string distname, ProxyID uuid, int callid, void** arr);
     
    ////////////
    // (Callee Method)
    // Waits to recieve all distributions necessary
    // before it returns the array. The notification that
    // a distribution has been received comes through the
    // doReceive() method.
    void* waitCompleteInArray(std::string distname, ProxyID uuid, int callid);

     
    ////////////
    // (Callee Method)
    // Waits to send all distributions necessary. The notification that
    // a distribution has been sent comes through the doSend() method.
    void waitCompleteOutArray(std::string distname, ProxyID uuid, int callid);
 
     ///////////
    // (Caller Method) 
    // It acquires all of the array descriptions that this object
    // needs to send part of its array to
    descriptorList* getRedistributionReps(std::string distname, ProxyID uuid);
 
    //////////
    // (Callee Method)
    // Retrieves a pointer to the ArrSynch, which provides access to the recieved
    // array as well as many other "great" synchronization methods
    MxNArrSynch* getArrSynch(::std::string distname, ProxyID uuid, int callid);

     //////////
    // (Callee Method)
    // Retrieves a pointer to the MetaSynch, which provides ynchronization methods
    // for setCallerDistribution and RMIs.
    MxNMetaSynch* getMetaSynch(::std::string distname, ProxyID uuid, int size);
   //////////
    // Erases the ScheduleEntry with a particular name
    void clear(std::string distname, ProxyID uuid, sched_t sch);

    //////////
    // Prints this object
    void print();

    //////////
    // Debug file
    std::ofstream dbg;

  private:
    ///////////
    // List of ScheduleEntries (the remote array represenations)
    // which represent every distibution that this object participates in.
    schedList entries;

    ///////////
    // List of array representations (the local array represenations)
    // which represent every distibution that this object participates in.
    arrRepList myreps;

    ////////
    // Mutex used to control access to MxNScheduleEntry::synchlist
    Mutex s_mutex;

    //TODO: change contructor.
    sched_t sch_type;
  };

} // End namespace SCIRun

#endif








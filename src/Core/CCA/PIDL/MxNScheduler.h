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

  ////////////////////
  // R = process rank; S = process size; L = array length in this dimension
  #define BLOCK(R,S,L) MxNScheduler::makeBlock(R,S,L)
  #define CYCLIC(R,S,L) MxNScheduler::makeCyclic(R,S,L)

  //////////////
  //A map of scheduleEntries to distibution names
  typedef std::map<std::string, MxNScheduleEntry*, ltstr> schedList;

  class MxNScheduler {    
  public:
    
    MxNScheduler();
    virtual ~MxNScheduler();

    ////////////
    // Accepts an array representation describing a callee/callee
    // and assigns it to the appropriate ScheduleEntry for that 
    // distribution name
    void setCalleeRepresentation(std::string distname, 
				 MxNArrayRep* arrrep);
    void setCallerRepresentation(std::string distname, 
				 MxNArrayRep* arrrep);
    
    ////////////
    // Checks wether we are playing a caller/callee method
    // from the perspective of the distribution and 
    // gets the native distribution representation
    MxNArrayRep* calleeGetCalleeRep(std::string distname);
    MxNArrayRep* callerGetCallerRep(std::string distname);

    /////////
    // Static methods that retrieve a appropriate Index
    // to each process participating in a defined distribution
    // such as block, cyclic, etc.
    static Index* makeBlock(int rank, int size, int length);
    static Index* makeCyclic(int rank, int size, int length);
    
    ///////////
    // (Callee Method)
    // Report the reception of array distribution metadata. Size denotes the
    // number of metadata receptions we are expecting.
    // See Also: MxNScheduleEntry::reportMetaRecvFinished(...)
    void reportMetaRecvDone(std::string distname, int size);

    ////////////
    // (Callee Metods)
    // These methods create a posibility to assign and access
    // an redistibuted array. This array is used throughout all
    // of the separate data receptions in order to eliminate 
    // copying
    // SeeAlso: similar methods in MxNScheduleEntry Class
    void setArray(std::string distname, std::string uuid, int callid, void** arr);
     
    ////////////
    // (Callee Method)
    // Waits to recieve all distributions necessary
    // before it returns the array. The notification that
    // a distribution has been received comes through the
    // reportRedisDone() method.
    void* waitCompleteArray(std::string distname, std::string uuid, int callid);
 
     ///////////
    // (Caller Method) 
    // It acquires all of the array descriptions that this object
    // needs to send part of its array to
    descriptorList getRedistributionReps(std::string distname);
 
    //////////
    // (Callee Method)
    // Retrieves a pointer to the ArrSynch, which provides access to the recieved
    // array as well as many other "great" synchronization methods
    MxNArrSynch* getArrSynch(::std::string distname, ::std::string uuid, int callid);

    //////////
    // Erases the ScheduleEntry with a particular name
    void clear(std::string distname);

    //////////
    // Prints this object
    void print();

    //////////
    // Debug file
    std::ofstream dbg;

  private:
    ///////////
    // List of ScheduleEntries which represent every distibution
    // that this object participates in.
    schedList entries;

    ////////
    // Mutex used to control access to MxNScheduleEntry::synchlist
    Mutex s_mutex;
    
  };

} // End namespace SCIRun

#endif








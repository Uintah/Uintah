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

#ifndef Component_PIDL_MxNScheduler_h
#define Component_PIDL_MxNScheduler_h

#include <string>
#include <map>
#include <fstream>
#include <Core/CCA/Component/PIDL/MxNArrayRep.h>
#include <Core/CCA/Component/PIDL/MxNScheduleEntry.h>

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
    
    ////////////
    // (Callee Metods)
    // These methods create a posibility to assign and access
    // an redistibuted array. This array is used throughout all
    // of the separate data receptions in order to eliminate 
    // copying
    // SeeAlso: similar methods in MxNScheduleEntry Class
    void* getArray(std::string distname);
    void* getArrayWait(std::string distname);
    void setArray(std::string distname, void** arr);
    void* getCompleteArray(std::string distname);
 
    ////////////
    // (Callee Method)
    // Marks a particular remote array description as received
    // when the actual data it describes has been received
    void markReceived(std::string distname,int rank);

    ///////////
    // (Callee Method)
    // Report the reception of array distribution metadata. Size denotes the
    // number of metadata receptions we are expecting.
    // See Also: MxNScheduleEntry::reportMetaRecvFinished(...)
    void reportMetaRecvFinished(std::string distname,int size); 

    ///////////
    // (Caller Method) 
    // It acquires all of the array descriptions that this object
    // needs to send part of its array to
    descriptorList getRedistributionReps(std::string distname);
 
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

  };

} // End namespace SCIRun

#endif








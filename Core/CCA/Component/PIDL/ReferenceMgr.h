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
 *  ReferenceMgr.h: Class which manages references to remote object. It is used
 *                    to enable parallel references and enable subset creation.
 *                      
 *                       .
 *                   
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef Component_PIDL_ReferenceMgr_h
#define Component_PIDL_ReferenceMgr_h

#include <Core/CCA/Component/PIDL/Reference.h>
#include <vector>

namespace SCIRun {
/**************************************
 
CLASS
   ReferenceMgr
   
KEYWORDS
   PIDL, Client, Proxy, Reference
   
DESCRIPTION
   The ReferenceMgr class is associated with a proxy object
   and it's purpose is to make manage the references to the
   remote component, regardless of whether this component is 
   parallel or serial. It also provides an interface to 
   facilitate the creation of subsets.

****************************************/
  typedef std::vector<Reference> refList;

  typedef enum {
    REDIS = 1,
    CALLONLY,
    CALLNORET,
    NOCALLRET
  } callType;
    

  class ReferenceMgr {
  public:
    
    /////////
    // Default constructor
    ReferenceMgr();

    /////////
    // Constructor with rank/size (parallel proxies)
    ReferenceMgr(int rank, int size);

    //////////
    // Copy the referenc manager.
    ReferenceMgr(const ReferenceMgr&);

    //////////
    // Copy the reference.manager
    ReferenceMgr& operator=(const ReferenceMgr&);

    //////////
    // Destructor which clears all current storage entries
    ~ReferenceMgr(); 

    //////////
    // Retreives a pointer to the reference needed for  
    // independent (if parallel) or serial RMI 
    Reference* getIndependentReference();

    //////////
    // Retreives a pointer to the reference needed for  
    // collective RMI
    ::std::vector<Reference*> getCollectiveReference(callType tip);

    /////////
    // Retreives a pointer to the list of all references
    refList* getAllReferences();

    //////////
    // Adds another reference (parallel component situation)
    void insertReference(const Reference& ref);

    ///////////
    // Returns the number of references in the list
    int getRemoteSize();

    /////////
    // For parallel proxies, number of cohorts
    int localSize;

    /////////
    // For parallel proxies, my ordered number
    int localRank;

    //////////
    // A vector of reference to the remote objects.
    refList d_ref;
  };
} // End namespace SCIRun

#endif





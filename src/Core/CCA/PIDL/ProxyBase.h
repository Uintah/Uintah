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
 *  ProxyBase.h: Base class for all PIDL proxies
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_ProxyBase_h
#define CCA_PIDL_ProxyBase_h

#include <Core/CCA/PIDL/Reference.h>
#include <Core/CCA/PIDL/ReferenceMgr.h>
#include <Core/CCA/PIDL/MxNScheduler.h>
#include <Core/CCA/PIDL/XceptionRelay.h>
#include <vector> 
#include <string>

namespace SCIRun {

  class XceptionRelay;

/**************************************
 
CLASS
   ProxyBase
   
KEYWORDS
   Proxy, PIDL
   
DESCRIPTION
   The base class for all proxy objects.  It contains the reference to
   the remote object (or references to all objects part of a parallel
   component).  This class should not be used outside of PIDL
   or automatically generated sidl code.
****************************************/
  class ProxyBase {
  public:
    
    //////////
    // A MxN Distribution Scheduler in case this proxy needs
    // to redistribute an array
    MxNScheduler* d_sched;
    
  protected:
    
    ProxyBase();
    
    ////////////
    // Create the proxy from the given reference.
    //remove it later
    ProxyBase(const Reference&);
    
    ////////////
    // Create the proxy from the given reference pointer.
    // the pointer is deleted inside the constructor
    ProxyBase(Reference*);
    
    ////////////
    // Create the proxy from the given reference list.
    ProxyBase(const ReferenceMgr&);	  
    
    ///////////
    // Destructor which closes connection
    virtual ~ProxyBase();
    
    //////////
    // Reference manager
    ReferenceMgr rm; 
    
    //////////
    // Xception relay which provides implementation of imprecise exceptions
    XceptionRelay* xr;
    
    //////////
    // TypeInfo is a friend so that it can call _proxyGetReference
    friend class TypeInfo;
    /////////
    // Xception is a friend since it needs access to the reference mgr's intracomm
    friend class XceptionRelay;
    
    //////////
    // Return a the first internal reference or a copy of it.  
    void _proxyGetReference(Reference& ref, bool copy) const;
    
    //////////
    // Returns the reference manager 
    ReferenceMgr* _proxyGetReferenceMgr() const;
    
    //////////
    // Returns the unique identifier of this proxy 
    ::std::string getProxyUUID(); 

    /////////
    // Created a subset of processes on the callee to service
    // all collective calls.
    void _proxycreateSubset(int localsize, int remotesize);

    /////////
    // Synchronizes parallel processes and throws all necessary exceptions
    void _proxygetException(); 

  private:
    
    ////////
    // A unique name representing the entire proxy component
    // (Mostly used to represent the same proxy among the proxy's 
    // parallel processes) 
    ::std::string proxy_uuid; 
  };
} // End namespace SCIRun

#endif






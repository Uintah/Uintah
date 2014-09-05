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

#ifndef Component_PIDL_ProxyBase_h
#define Component_PIDL_ProxyBase_h

#include <Core/CCA/Component/PIDL/Reference.h>
#include <Core/CCA/Component/PIDL/MxNScheduler.h>
#include <vector> 

namespace SCIRun {

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
        typedef std::vector<Reference> refList;

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
	  ProxyBase(const Reference&);

	  ////////////
	  // Create the proxy from the given reference list.
	  ProxyBase(const refList&);	  

	  ///////////
	  // Destructor which closes connection
	  virtual ~ProxyBase();
	  
	  //////////
	  // A vector of reference to the remote objects.
	  refList d_ref;
	  
	  //////////
	  // TypeInfo is a friend so that it can call _proxyGetReference
	  friend class TypeInfo;
	  
	  //////////
	  // Return a the first internal reference or a copy of it.  
	  void _proxyGetReference(Reference& ref, bool copy) const;

	  //////////
	  // Adds another reference (parallel component situation)  
	  void addParReference(Reference& ref);

	  /////////
	  // Returns the whole reference list or a copy of it.
	  void _proxyGetReferenceList(refList& ref, bool copy) const;
	  
	private:

	};
} // End namespace SCIRun

#endif






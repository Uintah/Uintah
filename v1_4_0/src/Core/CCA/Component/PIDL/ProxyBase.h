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

namespace PIDL {

/**************************************
 
CLASS
   ProxyBase
   
KEYWORDS
   Proxy, PIDL
   
DESCRIPTION
   The base class for all proxy objects.  It contains the reference to
   the remote object.  This class should not be used outside of PIDL
   or automatically generated sidl code.
****************************************/
	class ProxyBase {
	public:
	protected:
	    ////////////
	    // Create the proxy from the given reference.
	    ProxyBase(const Reference&);

	    ///////////
	    // Destructor
	    virtual ~ProxyBase();

	    //////////
	    // The reference to the remote object.
	    Reference d_ref;

	    //////////
	    // TypeInfo is a friend so that it can call _proxyGetReference
	    friend class TypeInfo;

	    //////////
	    // Return the internal reference.  If copy is true, the startpoint
	    // will be copied through globus_nexus_startpoint_copy, and
	    // will need to be destroyed with globus_nexus_startpoint_destroy
	    // or globus_nexus_put_startpoint_transfer.
	    void _proxyGetReference(Reference&, bool copy) const;

	    //////////
	    // Once the constructor of a sub class has finished
	    // successfully attached, it will call this function.
	    // This is used so that the destructor does NOT run if
	    // an attach was not already successful.
	    void attach_done() { attached_ = true; }

	private:
	    bool attached_;
	};
} // End namespace PIDL

#endif



/*
 *  Object.h: Base class for all PIDL distributed objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_PIDL_Object_proxy_h
#define Component_PIDL_Object_proxy_h

#include <Core/CCA/Component/PIDL/ProxyBase.h>
#include <Core/CCA/Component/PIDL/Object.h>

namespace PIDL {
/**************************************
 
CLASS
   Object_proxy
   
KEYWORDS
   Proxy, Object, PIDL
   
DESCRIPTION
   Internal PIDL class for a proxy to a base object.  This impements
   the Object_interface interface and provides a proxy mechanism for
   remote objects.  Since there are no interesting methods at this level,
   the only interesting thing that we can do is up-cast.
****************************************/
	class Object_proxy : public ProxyBase, public Object_interface {
	public:
	protected:
	    //////////
	    // PIDL will create these.
	    friend class PIDL;

	    //////////
	    // Private constructor from a reference
	    Object_proxy(const Reference&);

	    //////////
	    // Private constructor from a URL
	    Object_proxy(const URL&);

	    //////////
	    // Destructor
	    virtual ~Object_proxy();
	private:
	};
} // End namespace PIDL

#endif


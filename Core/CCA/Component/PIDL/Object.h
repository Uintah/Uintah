
/*
 *  Object.h: Base class for all PIDL distributed objects
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_PIDL_Object_h
#define Component_PIDL_Object_h

namespace Component {
    namespace PIDL {
	class Reference;
	class ServerContext;
	class TypeInfo;
	class URL;

/**************************************
 
CLASS
   Object_interface
   
KEYWORDS
   Object, PIDL
   
DESCRIPTION
   The base class for all PIDL based distributed objects.  It provides
   a single public method - getURL, which returns a URL for that object.
   The _getReference method is also in the public section, but should
   be considered private to the implementation and should not be used
   outside of PIDL or sidl generated stubs.

****************************************/
	class Object_interface {
	public:
	    //////////
	    // Destructor
	    virtual ~Object_interface();

	    //////////
	    // Return a URL that can be used to contact this object.
	    URL getURL() const;

	    //////////
	    // Internal method to get a reference (startpoint and
	    // vtable base) to this object.
	    virtual void _getReference(Reference&, bool copy) const;
	protected:
	    //////////
	    // Constructor.  Initializes d_serverContext to null,
	    // which means that it is a proxy object.  Calls to
	    // intializeServer from derived class constructors will
	    // set up d_serverContext appropriately.
	    Object_interface();

	    //////////
	    // Set up d_serverContext with the given type information
	    // and the void pointer to be passed back to the generated
	    // stub code.  This will get called by each constructor
	    // in the entire inheritance tree.  The last one to call
	    // will be the most derived class, which is the only
	    // one that we care about.
	    void initializeServer(const TypeInfo* typeinfo, void* ptr);
	private:
	    friend class Wharehouse;
	    //////////
	    // The context of the server object.  If this is null,
	    // then the object is a proxy and there is no server.
	    ServerContext* d_serverContext;

	    //////////
	    // Activate the server.  This registers the object with the
	    // object wharehouse, and creates the globus endpoint.
	    // This is not done until something requests a reference
	    // to the object though getURL() or _getReference().
	    void activateObject() const;

	    //////////
	    // Private copy constructor to make copying impossible.
	    Object_interface(const Object_interface&);

	    //////////
	    // Private assignment operator to make assignment impossible.
	    Object_interface& operator=(const Object_interface&);
	};

/**************************************
 
CLASS
   Object
   
KEYWORDS
   Handle, PIDL
   
DESCRIPTION
   A pointer to the base object class.  Will somebody be replaced
   with a a smart pointer class.

****************************************/
	typedef Object_interface* Object;
    }
}

#endif

//
// $Log$
// Revision 1.3  1999/09/24 06:26:25  sparker
// Further implementation of new Component model and IDL parser, including:
//  - fixed bugs in multiple inheritance
//  - added test for multiple inheritance
//  - fixed bugs in object reference send/receive
//  - added test for sending objects
//  - beginnings of support for separate compilation of sidl files
//  - beginnings of CIA spec implementation
//  - beginnings of cocoon docs in PIDL
//  - cleaned up initalization sequence of server objects
//  - use globus_nexus_startpoint_eventually_destroy (contained in
// 	the globus-1.1-utah.patch)
//
// Revision 1.2  1999/09/17 05:08:08  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.1  1999/08/30 17:39:46  sparker
// Updates to configure script:
//  rebuild configure if configure.in changes (Bug #35)
//  Fixed rule for rebuilding Makefile from Makefile.in (Bug #36)
//  Rerun configure if configure changes (Bug #37)
//  Don't build Makefiles for modules that aren't --enabled (Bug #49)
//  Updated Makfiles to build sidl and Component if --enable-parallel
// Updates to sidl code to compile on linux
// Imported PIDL code
// Created top-level Component directory
// Added ProcessManager class - a simpler interface to fork/exec (not finished)
//
//

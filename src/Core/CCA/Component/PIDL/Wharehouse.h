
/*
 *  Wharehouse.h: A pile of distributed objects
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

#ifndef Component_PIDL_Wharehouse_h
#define Component_PIDL_Wharehouse_h

#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/Mutex.h>
#include <map>
#include <string>
#include <globus_nexus.h>

namespace Component {
    namespace PIDL {
	class Object_interface;

/**************************************
 
CLASS
   Wharehouse
   
KEYWORDS
   Wharehouse, PIDL
   
DESCRIPTION
   Internal PIDL class. This is a singleton that holds all activated
   server objects.
****************************************/
	class Wharehouse {
	public:
	    //////////
	    // The nexus approval function.  Returns the startpoint
	    // to an object based on the object number.
	    int approval(char* url, globus_nexus_startpoint_t* sp);

	protected:
	    //////////
	    // PIDL needs access to most of these methods.
	    friend class PIDL;

	    //////////
	    // The constructor - only called once.
	    Wharehouse();

	    //////////
	    // Destructor
	    ~Wharehouse();

	    //////////
	    // The Object base class will register server objects with
	    // the wharehouse.
	    friend class Object_interface;

	    //////////
	    // Register obj with the wharehouse, returning the objects
	    // unique identifier.
	    int registerObject(Object_interface* obj);

	    //////////
	    // Unregister the object associated with the object ID.
	    // Returns a pointer to the object.
	    Object_interface* unregisterObject(int id);

	    //////////
	    // Lookup an object by name.  name should be parsable
	    // as an integer, specifiying the object id.  Returns
	    // null of the object is not found.  May throw
	    // InvalidReference if name is not parsable.
	    Object_interface* lookupObject(const std::string&);

	    //////////
	    // Lookup an object by the object ID.  Returns null if
	    // the object is not found.
	    Object_interface* lookupObject(int id);

	    //////////
	    // "Run" the wharehouse.  This simply blocks until objects
	    // have been removed from the wharehouse.
	    void run();

	private:
	    //////////
	    // The lock for the object database and nextID
	    SCICore::Thread::Mutex mutex;

	    //////////
	    // The wait condition for run().  It is signaled when all
	    // objects have been removed from the wharehouse.
	    SCICore::Thread::ConditionVariable condition;

	    //////////
	    // The object database
	    std::map<int, Object_interface*> objects;

	    //////////
	    // The ID of the next object to be created.
	    int nextID;
	};
    }
}

#endif

//
// $Log$
// Revision 1.4  1999/09/24 20:03:39  sparker
// Added cocoon documentation
//
// Revision 1.3  1999/09/17 05:08:11  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:03  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:49  sparker
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

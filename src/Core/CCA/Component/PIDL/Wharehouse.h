
/*
 *  Wharehouse.h: A pile of distributed objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core/CCA/Component_PIDL_Wharehouse_h
#define Core/CCA/Component_PIDL_Wharehouse_h

#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Mutex.h>
#include <map>
#include <string>
#include <globus_nexus.h>

namespace SCIRun {

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
	    Mutex mutex;

	    //////////
	    // The wait condition for run().  It is signaled when all
	    // objects have been removed from the wharehouse.
	    ConditionVariable condition;

	    //////////
	    // The object database
	    std::map<int, Object_interface*> objects;

	    //////////
	    // The ID of the next object to be created.
	    int nextID;
	};
} // End namespace SCIRun

#endif


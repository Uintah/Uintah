
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

	class Wharehouse {
	public:
	    int approval(char* url, globus_nexus_startpoint_t* sp);
	protected:
	    friend class PIDL;

	    Wharehouse();
	    ~Wharehouse();

	    friend class Object_interface;
	    int registerObject(Object_interface*);
	    Object_interface* unregisterObject(int);
	    Object_interface* lookupObject(const std::string&);
	    Object_interface* lookupObject(int id);
	    void run();

	private:
	    SCICore::Thread::Mutex mutex;
	    SCICore::Thread::ConditionVariable condition;
	    std::map<int, Object_interface*> objects;
	    int nextID;
	};
    }
}

#endif

//
// $Log$
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

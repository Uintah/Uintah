
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

	class Object_interface {
	public:
	    virtual ~Object_interface();
	    URL getURL() const;

	    virtual void _getReference(Reference&, bool copy) const;
	protected:
	    Object_interface();
	    void initializeServer(const TypeInfo* typeinfo, void* ptr);
	private:
	    friend class Wharehouse;
	    ServerContext* d_serverContext;

	    Object_interface(const Object_interface&);
	    Object_interface& operator=(const Object_interface&);
	};
	typedef Object_interface* Object;
    }
}

#endif

//
// $Log$
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

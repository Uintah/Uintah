
/*
 *  PIDL.h: Include a bunch of PIDL files for external clients
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core/CCA/Component_PIDL_PIDL_h
#define Core/CCA/Component_PIDL_PIDL_h

#include <Core/CCA/Component/PIDL/Object.h>
#include <Core/CCA/Component/PIDL/PIDLException.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <Core/CCA/Component/PIDL/pidl_cast.h>
#include <string>

namespace SCIRun {

/**************************************
 
CLASS
   PIDL
   
KEYWORDS
   PIDL
   
DESCRIPTION
   A class to encapsulate several static methods for PIDL.
****************************************/
	class PIDL {
	public:
	    //////////
	    // Initialize PIDL
	    static void initialize(int argc, char* argv[]);

	    //////////
	    // Create a base Object class from the given URL
	    static Object objectFrom(const URL&);

	    //////////
	    // Create a base Object class from the given Reference
	    static Object objectFrom(const Reference&);

	    //////////
	    // Go into the main loop which services requests for all
	    // objects.  This will not return until all objects have
	    // been destroyed.
	    static void serveObjects();

	    //////////
	    // Return the URL for the current process.  Individual
	    // objects may be identified by appending their id number
	    // or name to the end of the string.
	    static std::string getBaseURL();

	    //////////
	    // Return the object Wharehouse.  Most clients will not
	    // need to use this.
	    static Wharehouse* getWharehouse();
	protected:
	private:
	    //////////
	    // The wharehouse singleton object
	    static Wharehouse* wharehouse;

	    //////////
	    // Private constructor to prevent creation of a PIDL
	    PIDL();
	};
} // End namespace SCIRun

#endif


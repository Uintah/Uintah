
/*
 *  Time: Utility class for dealing with time
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_Time_h
#define SCICore_Thread_Time_h

#include <SCICore/share/share.h>

namespace SCICore {
    namespace Thread {
/**************************************
 
CLASS
   Time
   
KEYWORDS
   Thread, Time
   
DESCRIPTION
   Utility class to manage Time.  This class is implemented using
   high precision counters on the SGI, and standard unix system calls
   on other machines.

****************************************/
	class SCICORESHARE Time {
	public:
	    typedef unsigned long long SysClock;
	    
	    //////////
	    // Return the current system time, in terms of clock ticks.
	    // Time zero is at some arbitrary point in the past.
	    static SysClock currentTicks();
	    
	    //////////
	    // Return the current system time, in terms of seconds.
	    // This is slower than currentTicks().  Time zero is at
	    // some arbitrary point in the past.
	    static double currentSeconds();
	    
	    //////////
	    // Return the conversion from seconds to ticks.
	    static double ticksPerSecond();
	    
	    //////////
	    // Return the conversion from ticks to seconds.
	    static double secondsPerTick();
	    
	    //////////
	    // Wait until the specified time in clock ticks.
	    static void waitUntil(SysClock ticks);
	    
	    //////////
	    // Wait until the specified time in seconds.
	    static void waitUntil(double seconds);
	    
	    //////////
	    // Wait for the specified time in clock ticks
	    static void waitFor(SysClock ticks);
	    
	    //////////
	    // Wait for the specified time in seconds
	    static void waitFor(double seconds);

	private:
	    Time();
	    static void initialize();
	};
    }
}

#endif

//
// $Log$
// Revision 1.6  1999/09/24 18:55:08  moulding
// added SCICORESHARE, for win32, to class declarations
//
// Revision 1.5  1999/09/02 16:52:44  sparker
// Updates to cocoon documentation
//
// Revision 1.4  1999/08/28 03:46:52  sparker
// Final updates before integration with PSE
//
// Revision 1.3  1999/08/25 19:00:53  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.2  1999/08/25 02:38:03  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//


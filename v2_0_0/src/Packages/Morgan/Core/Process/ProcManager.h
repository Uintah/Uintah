
/*
 *  ProcManager.h: Manage forking/rshing of processes.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Updated by:
 *   Jason V. Morgan
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Morgan_Process_ProcManager_h
#define Morgan_Process_ProcManager_h

#include <Packages/Morgan/Core/Process/ProcManagerException.h>
#include <string>

namespace Morgan {
    namespace Process {
    class Proc;
	class ProcManager {
	public:
	    static Proc* start_proc(const char* command, ...);
	protected:
	private:
	    ProcManager();
	    ~ProcManager();
	};
    }
}

#endif


/*
 *  ProcessManager.h: Manage forking/rshing of processes.
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Process_ProcessManager_h
#define Core_Process_ProcessManager_h

#include <Core/Process/ProcessManagerException.h>
#include <string>

namespace SCIRun {
	class ProcessManager {
	public:
	    static void start_process(const std::string& command,
				      const std::string& resourceSpec);
	protected:
	private:
	    ProcessManager();
	    ~ProcessManager();
	};
} // End namespace SCIRun

#endif


/*
 *  ProcessManagerException.h: Base class for Process Manager
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Process_ProcessManagerException_h
#define Core_Process_ProcessManagerException_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
	class ProcessManagerException : public Exception {
	public:
	    ProcessManagerException(const ProcessManagerException&);
	protected:
	private:
	};
} // End namespace SCIRun

#endif

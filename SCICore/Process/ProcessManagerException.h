
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

#ifndef SCICore_Process_ProcessManagerException_h
#define SCICore_Process_ProcessManagerException_h

#include <SCICore/Exceptions/Exception.h>

namespace SCICore {
    namespace Process {
	class ProcessManagerException : public SCICore::Exceptions::Exception {
	public:
	protected:
	private:
	};
    }
}

#endif

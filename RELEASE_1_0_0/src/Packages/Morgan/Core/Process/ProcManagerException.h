
/*
 *  ProcManagerException.h: Process Manager Exception class
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

#ifndef Morgan_Process_ProcManagerException_h
#define Morgan_Process_ProcManagerException_h

#include <Core/Exceptions/Exception.h>

namespace Morgan {
    namespace Process {
	class ProcManagerException : public SCIRun::Exception {
	public:
        ProcManagerException(const char* imsg) : msg(imsg) {};
        virtual const char* message() const { return msg; }
        virtual const char* type() const 
            { return "Morgan::Process::ProcManagerException"; }
	protected:
	private:
        const char* msg;
	};
    }
}

#endif


/*
 *  PIDLException.h: Base class for PIDL Exceptions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core/CCA/Component_PIDL_PIDLException_h
#define Core/CCA/Component_PIDL_PIDLException_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
/**************************************
 
CLASS
   PIDLException
   
KEYWORDS
   PIDL, Exception
   
DESCRIPTION
   The base class for all PIDL exceptions.  This provides a convenient
   mechanism for catch all PIDL exceptions.  It is abstract because
   Exception is abstract, so cannot be instantiated.
   It provides no additional methods beyond the Core base exception
   class.
****************************************/
	class PIDLException : public Exception {
	public:
	    PIDLException();
	    PIDLException(const PIDLException&);
	protected:
	private:
	    PIDLException& operator=(const PIDLException&);
	};
} // End namespace SCIRun

#endif



// $Id$

/*
 *  FileNotFound.h: Exactly what you think
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCICore_Exceptions_FileNotFound_h
#define SCICore_Exceptions_FileNotFound_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace SCICore {
    namespace Exceptions {
	class FileNotFound : public Exception {
	public:
	    FileNotFound(const std::string&);
	    FileNotFound(const FileNotFound&);
	    virtual ~FileNotFound();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    std::string d_message;
	    FileNotFound& operator=(const FileNotFound&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  2000/05/20 08:06:14  sparker
// New exception classes
//
//


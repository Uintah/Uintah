

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

#ifndef Core_Exceptions_FileNotFound_h
#define Core_Exceptions_FileNotFound_h

#include <Core/Exceptions/Exception.h>
#include <string>

namespace SCIRun {
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
} // End namespace SCIRun

#endif



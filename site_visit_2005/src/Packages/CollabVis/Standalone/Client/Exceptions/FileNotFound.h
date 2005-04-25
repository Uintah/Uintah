/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/



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
	    std::string message_;
	    FileNotFound& operator=(const FileNotFound&);
	};
} // End namespace SCIRun

#endif



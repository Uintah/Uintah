/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



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

#ifndef CCA_PIDL_PIDLException_h
#define CCA_PIDL_PIDLException_h

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
	class PIDLException : public SCIRun::Exception {
	public:
	    PIDLException();
	    PIDLException(const PIDLException&);
	protected:
	private:
	    PIDLException& operator=(const PIDLException&);
	};
} // End namespace SCIRun

#endif


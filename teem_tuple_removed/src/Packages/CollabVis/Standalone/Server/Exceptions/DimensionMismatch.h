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
 *  DimensionMismatch.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_DimensionMismatch_h
#define Core_Exceptions_DimensionMismatch_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
	class DimensionMismatch : public Exception {
	public:
	    DimensionMismatch(long value, long expected);
	    DimensionMismatch(const DimensionMismatch&);
	    virtual ~DimensionMismatch();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    long value, expected;
	    char* msg;

	    DimensionMismatch& operator=(const DimensionMismatch);
	};
} // End namespace SCIRun

#endif  // Core_Exceptions_DimensionMismatch_h



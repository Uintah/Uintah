/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  ArrayIndexOutOfBounds.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 */

#ifndef Core_Exceptions_ArrayIndexOutOfBounds_h
#define Core_Exceptions_ArrayIndexOutOfBounds_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
	class ArrayIndexOutOfBounds : public Exception {
	public:
	    ArrayIndexOutOfBounds(long value, long lower, long upper, 
                                  const char* file, int line);
	    ArrayIndexOutOfBounds(const ArrayIndexOutOfBounds&);
	    virtual ~ArrayIndexOutOfBounds();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    long value, lower, upper;
	    char* msg;

	    ArrayIndexOutOfBounds& operator=(const ArrayIndexOutOfBounds);
	};
} // End namespace SCIRun

#endif



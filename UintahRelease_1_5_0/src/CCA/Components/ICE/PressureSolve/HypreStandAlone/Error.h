/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*#############################################################################
  # Error.h - error messages handler
  #============================================================================
  # Error serves as a base class from which most of the other data classes
  # inherits (Residual, Equation, Function, etc). It is a handler that outputs
  # an error messages and sends an exception. It can of course be overriden to
  # be specialized by some of the mentioned classes.
  #
  # See also: Macros.
  #
  # Revision history:
  # -----------------
  # 07-DEC-2004   Added comments.
  # 12-JAN-2005   Changed included headers, added warning().
  ###########################################################################*/

#ifndef _ERROR_H
#define _ERROR_H

#include "Macros.h"

/*============= Begin class Error =============*/

class Error {
public:
  
  /*------------- Construction, destruction, copy -------------*/

  Error(void) {}
  virtual ~Error(void) {}

  /*------------- Error handling -------------*/

  virtual std::string summary(void) const;
  virtual void error(const std::ostringstream& msg) const;
  virtual void warning(const std::ostringstream& msg) const;
};

/*============= End class Error =============*/

#endif /* _ERROR_H */

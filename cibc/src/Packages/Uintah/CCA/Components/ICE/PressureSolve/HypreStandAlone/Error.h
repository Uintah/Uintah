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

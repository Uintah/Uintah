

/*
 *  AssertionFailed.h: Exception for a failed assertion.  Note - this
 *    version takes only a char*.  There is a FancyAssertionFailed that
 *    takes std::string.  This is done to prevent include file pollution
 *    with std::string
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_AssertionFailed_h
#define Core_Exceptions_AssertionFailed_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
class AssertionFailed : public Exception {
public:
  AssertionFailed(const char* msg,
		  const char* file,
		  int line);
  AssertionFailed(const AssertionFailed&);
  virtual ~AssertionFailed();
  virtual const char* message() const;
  virtual const char* type() const;
protected:
private:
  char* d_message;
  AssertionFailed& operator=(const AssertionFailed&);
};
} // End namespace SCIRun

#endif



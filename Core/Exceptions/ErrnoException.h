/*
 *  ErrnoException.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Core_Exceptions_ErrnoException_h
#define Core_Exceptions_ErrnoException_h

#include <Core/Exceptions/Exception.h>
#include <string>

namespace SCIRun {

class ErrnoException : public Exception {
public:
  ErrnoException(const std::string&, int err);
  ErrnoException(const ErrnoException&);
  virtual ~ErrnoException();
  virtual const char* message() const;
  virtual const char* type() const;
	 
  int getErrno() const;
protected:
private:
  std::string message_;
  int errno_;

  ErrnoException& operator=(const ErrnoException&);
};

} // End namespace SCIRun

#endif

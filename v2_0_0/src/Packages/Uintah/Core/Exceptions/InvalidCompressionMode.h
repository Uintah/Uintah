/*
 *  InvalidCompressionMode.h: 
 *
 *  Written by:
 *   Wayne Witzel
 *   Department of Computer Science
 *   University of Utah
 *   Febuary 2001
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_INVALIDCOMPRESSIONMODE_H
#define UINTAH_EXCEPTIONS_INVALIDCOMPRESSIONMODE_H

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using namespace SCIRun;

  class InvalidCompressionMode : public Exception {
  public:
    InvalidCompressionMode(const std::string& invalidmode,
			   const std::string& vartype="");
    InvalidCompressionMode(const InvalidCompressionMode&);
    virtual ~InvalidCompressionMode();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string d_msg;
    InvalidCompressionMode& operator=(const InvalidCompressionMode&);
  };
  
} // End namespace Uintah

#endif

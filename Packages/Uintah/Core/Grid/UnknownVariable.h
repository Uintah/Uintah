
/*
 *  UnknownVariable.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_UNKNOWNVARIABLE_H
#define UINTAH_EXCEPTIONS_UNKNOWNVARIABLE_H

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using SCIRun::Exception;

  class Level; 
  class Patch;

  class UnknownVariable : public Exception {
  public:
    UnknownVariable(const std::string& varname, int dwid, const Patch* patch,
		    int matlIndex, const std::string& extramsg = "");
    UnknownVariable(const std::string& varname, int dwid, const Level* level,
		    int matlIndex, const std::string& extramsg = "");
    UnknownVariable(const UnknownVariable&);
    virtual ~UnknownVariable();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string d_msg;
    UnknownVariable& operator=(const UnknownVariable&);
  };
} // End namespace Uintah

#endif

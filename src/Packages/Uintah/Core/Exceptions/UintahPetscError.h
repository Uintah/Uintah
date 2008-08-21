
/*
 *  UintahPetscError.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef UINTAH_EXCEPTIONS_PETSCERROR_H
#define UINTAH_EXCEPTIONS_PETSCERROR_H

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using namespace SCIRun;
  class UintahPetscError : public Exception {
  public:
    UintahPetscError(int petsc_code, const std::string&, const char* file, int line);
    UintahPetscError(const UintahPetscError&);
    virtual ~UintahPetscError();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    int petsc_code;
    std::string d_msg;
    UintahPetscError& operator=(const UintahPetscError&);
  };
} // End namespace Uintah

#endif

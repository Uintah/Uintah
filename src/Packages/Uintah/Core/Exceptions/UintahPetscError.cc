
#include <Packages/Uintah/Core/Exceptions/UintahPetscError.h>

#include <sgi_stl_warnings_off.h>
#include   <sstream>
#include <sgi_stl_warnings_on.h>

#include <iostream>

using namespace Uintah;


UintahPetscError::UintahPetscError(int petsc_code, const std::string& msg, const char* file, int line)
    : petsc_code(petsc_code)
{
  std::ostringstream out;
  out << "A UintahPetscError exception was thrown.\n"
      << file << ":" << line << "\n"
      << "PETSc error: " << petsc_code << ", " << msg;
  d_msg = out.str();
  
#ifdef EXCEPTIONS_CRASH
  cout << d_msg << "\n";
#endif  
}

UintahPetscError::UintahPetscError(const UintahPetscError& copy)
    : petsc_code(copy.petsc_code), d_msg(copy.d_msg)
{
}

UintahPetscError::~UintahPetscError()
{
}

const char* UintahPetscError::message() const
{
    return d_msg.c_str();
}

const char* UintahPetscError::type() const
{
    return "Uintah::Exceptions::UintahPetscError";
}


#include <Packages/Uintah/Core/Exceptions/PetscError.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace std;

PetscError::PetscError(int petsc_code, const std::string& msg)
    : petsc_code(petsc_code)
{
  ostringstream out;
  out << "PETSc error: " << petsc_code << ", " << msg;
  d_msg = out.str();
}

PetscError::PetscError(const PetscError& copy)
    : petsc_code(copy.petsc_code), d_msg(copy.d_msg)
{
}

PetscError::~PetscError()
{
}

const char* PetscError::message() const
{
    return d_msg.c_str();
}

const char* PetscError::type() const
{
    return "Uintah::Exceptions::PetscError";
}


#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <iostream>
using namespace Uintah;

ParameterNotFound::ParameterNotFound(const std::string& msg)
    : ProblemSetupException("Required parameter not found: "+msg)
{
#ifdef EXCEPTIONS_CRASH
  std::cout << msg << "\n";
#endif
}

ParameterNotFound::ParameterNotFound(const ParameterNotFound& copy)
    : ProblemSetupException(copy)
{
}

ParameterNotFound::~ParameterNotFound()
{
}

const char* ParameterNotFound::type() const
{
    return "Uintah::Exceptions::ParameterNotFound";
}

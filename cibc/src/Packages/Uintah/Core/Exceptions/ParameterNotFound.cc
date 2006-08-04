
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <iostream>
using namespace Uintah;

ParameterNotFound::ParameterNotFound(const std::string& msg, const char* file, int line)
    : ProblemSetupException("Required parameter not found: "+msg, __FILE__, __LINE__)
{
#ifdef EXCEPTIONS_CRASH
  std::cout << "A ProblemNotFound exception was thrown.\n";
  std::cout << file << ":" << line << "\n";
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


#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>

using namespace Uintah;

ParameterNotFound::ParameterNotFound(const std::string& msg)
    : ProblemSetupException("Required parameter not found: "+msg)
{
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


#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

using namespace Uintah;

ProblemSetupException::ProblemSetupException(const std::string& msg)
    : d_msg(msg)
{
}

ProblemSetupException::ProblemSetupException(const ProblemSetupException& copy)
    : d_msg(copy.d_msg)
{
}

ProblemSetupException::~ProblemSetupException()
{
}

const char* ProblemSetupException::message() const
{
    return d_msg.c_str();
}

const char* ProblemSetupException::type() const
{
    return "Uintah::Exceptions::ProblemSetupException";
}

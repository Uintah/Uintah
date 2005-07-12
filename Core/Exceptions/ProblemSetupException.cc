
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <iostream>

using namespace Uintah;

ProblemSetupException::ProblemSetupException(const std::string& msg, const char* file, int line)
    : d_msg(msg)
{
#ifdef EXCEPTIONS_CRASH
  std::cout << "A ProblemSetup exception was thrown.\n";
  std::cout << file << ":" << line << "\n";
  std::cout << d_msg << "\n";
#endif
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

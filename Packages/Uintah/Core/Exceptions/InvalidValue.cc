
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>

using namespace Uintah;

InvalidValue::InvalidValue(const std::string& msg, const char* file, int line)
    : ProblemSetupException(msg, __FILE__, __LINE__)
{
#ifdef EXCEPTIONS_CRASH
  std::cout << "An Invalid Value exception was thrown.\n";
  std::cout << file << ":" << line << "\n";
  std::cout << msg << "\n";
#endif
}

InvalidValue::InvalidValue(const InvalidValue& copy)
    : ProblemSetupException(copy)
{
}

InvalidValue::~InvalidValue()
{
}

const char* InvalidValue::type() const
{
    return "Uintah::Exceptions::InvalidValue";
}


#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>

using namespace Uintah;

InvalidValue::InvalidValue(const std::string& msg)
    : ProblemSetupException(msg)
{
#ifdef EXCEPTIONS_CRASH
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

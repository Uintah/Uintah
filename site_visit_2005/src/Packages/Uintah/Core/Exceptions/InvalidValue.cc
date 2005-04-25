
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

using namespace Uintah;

InvalidValue::InvalidValue(const std::string& msg)
    : ProblemSetupException(msg)
{
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

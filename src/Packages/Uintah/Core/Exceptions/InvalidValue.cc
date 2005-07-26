
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>

using namespace Uintah;

InvalidValue::InvalidValue(const std::string& msg, const char* file, int line)
  : d_msg(msg)
{
#ifdef EXCEPTIONS_CRASH
  std::cout << "An InvalidValue exception was thrown.\n";
  std::cout << file << ":" << line << "\n";
  std::cout << msg << "\n";
#endif
}

InvalidValue::InvalidValue(const InvalidValue& copy)
  : d_msg(copy.d_msg)
{
}

InvalidValue::~InvalidValue()
{
}

const char* InvalidValue::message() const
{
    return d_msg.c_str();
}

const char* InvalidValue::type() const
{
    return "Uintah::Exceptions::InvalidValue";
}

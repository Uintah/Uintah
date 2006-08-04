
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <iostream>
#include <sstream>

using namespace Uintah;
using std::cout;

InvalidValue::InvalidValue(const std::string& msg, const char* file, int line)
  : d_msg(msg)
{
  std::ostringstream s;
  s  << "An InvalidValue exception was thrown.\n" 
     << file << ":" << line << "\n" << d_msg;
  d_msg = s.str();
  
#ifdef EXCEPTIONS_CRASH
  std::cout << d_msg << "\n";
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

#include <Packages/Uintah/Core/Exceptions/InvalidCompressionMode.h>

using namespace Uintah;
using namespace std;

InvalidCompressionMode::InvalidCompressionMode(const std::string& invalidmode,
					       const std::string& vartype)
{
  d_msg = string("'") + invalidmode +
    string("' is not a valid compression mode");

  if (vartype != "") {
    d_msg += string(" for a variable of type ") + vartype;
  }
}

InvalidCompressionMode::InvalidCompressionMode(const InvalidCompressionMode& copy)
    : d_msg(copy.d_msg)
{
}

InvalidCompressionMode::~InvalidCompressionMode()
{
}

const char* InvalidCompressionMode::message() const
{
  return d_msg.c_str();
}

const char* InvalidCompressionMode::type() const
{
  return "Packages/Uintah::Exceptions::InvalidCompressionMode";
}

#include <Packages/Uintah/Core/Exceptions/InvalidCompressionMode.h>
#include <iostream>
#include <sstream>
using namespace Uintah;
using namespace std;

InvalidCompressionMode::InvalidCompressionMode(const std::string& invalidmode,
					       const std::string& vartype,
                                               const char* file,
                                               int line)
{
  ostringstream s;
  s << "An InvalidCompressionMode exception was thrown.\n" 
    << file << ":" << line << "\n"
    << "'" << invalidmode << "' is not a valid compression mode";
  d_msg = s.str();

  if (vartype != "") {
    d_msg += string(" for a variable of type ") + vartype;
  }
  
#ifdef EXCEPTIONS_CRASH
  cout << d_msg << "\n";
#endif
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

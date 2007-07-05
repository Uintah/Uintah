// license

#include <Packages/Kepler/Core/Exceptions/ServerError.h>

#include <iostream>
#include <sstream>

namespace Kepler {

ServerError::ServerError(const std::string& message, const char* file, int line)
    : message_(message)
{
  std::ostringstream s;
  s << "A Kepler::ServerError exception was thrown in\n"
    << file << " at " << line << "\n" << message_;
  message_ = (char*)(s.str().c_str());

#ifdef EXCEPTIONS_CRASH
  std::cout << message_ << "\n";
#endif
}

ServerError::ServerError(const ServerError& copy)
    : message_(copy.message_)
{
}

ServerError::~ServerError()
{
}

const char* ServerError::message() const
{
    return message_.c_str();
}

const char* ServerError::type() const
{
    return "ServerError";
}

}

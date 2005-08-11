
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <iostream>
#include <sstream>

using namespace Uintah;

TypeMismatchException::TypeMismatchException(const std::string& msg, const char* file, int line)
    : d_msg(msg)
{
  std::ostringstream s;
  s << "A TypeMismatchException was thrown.\n"
    << file << ":" << line << "\n";
  d_msg = s.str();

#ifdef EXCEPTIONS_CRASH
  std::cout << d_msg << "\n";
#endif
}

TypeMismatchException::TypeMismatchException(const TypeMismatchException& copy)
    : d_msg(copy.d_msg)
{
}

TypeMismatchException::~TypeMismatchException()
{
}

const char* TypeMismatchException::message() const
{
    return d_msg.c_str();
}

const char* TypeMismatchException::type() const
{
    return "Uintah::Exceptions::TypeMismatchException";
}

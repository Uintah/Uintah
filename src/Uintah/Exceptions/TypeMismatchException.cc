
#include "TypeMismatchException.h"

TypeMismatchException::TypeMismatchException(const std::string& msg)
    : msg(msg)
{
}

TypeMismatchException::TypeMismatchException(const TypeMismatchException& copy)
    : msg(copy.msg)
{
}

const char* TypeMismatchException::message() const
{
    return msg.c_str();
}

const char* TypeMismatchException::type() const
{
    return "Uintah::Exceptions::TypeMismatchException";
}

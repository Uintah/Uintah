
#include "TypeMismatchException.h"

TypeMismatchException::TypeMismatchException(const std::string& msg)
    : msg(msg)
{
}

std::string TypeMismatchException::message() const
{
    return msg;
}

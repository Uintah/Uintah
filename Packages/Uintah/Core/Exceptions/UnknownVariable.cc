
#include "UnknownVariable.h"

using namespace Uintah;

UnknownVariable::UnknownVariable(const std::string& msg)
    : d_msg(msg)
{
}

UnknownVariable::UnknownVariable(const UnknownVariable& copy)
    : d_msg(copy.d_msg)
{
}

UnknownVariable::~UnknownVariable()
{
}

const char* UnknownVariable::message() const
{
    return d_msg.c_str();
}

const char* UnknownVariable::type() const
{
    return "Uintah::Exceptions::UnknownVariable";
}

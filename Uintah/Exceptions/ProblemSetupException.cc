
#include "ProblemSetupException.h"

ProblemSetupException::ProblemSetupException(const std::string& msg)
    : msg(msg)
{
}

const char* ProblemSetupException::message() const
{
    return msg.c_str();
}

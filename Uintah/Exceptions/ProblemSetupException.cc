
#include "ProblemSetupException.h"

ProblemSetupException::ProblemSetupException(const std::string& msg)
    : msg(msg)
{
}

std::string ProblemSetupException::message() const
{
    return msg;
}

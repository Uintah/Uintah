
#include "ProblemSetupException.h"

ProblemSetupException::ProblemSetupException(const std::string& msg)
    : msg(msg)
{
}

ProblemSetupException::ProblemSetupException(const ProblemSetupException& copy)
    : msg(copy.msg)
{
}

const char* ProblemSetupException::message() const
{
    return msg.c_str();
}

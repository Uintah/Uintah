
#include "SchedulerException.h"

SchedulerException::SchedulerException(const std::string& msg)
    : msg(msg)
{
}

std::string SchedulerException::message() const
{
    return msg;
}


#include "SchedulerException.h"

SchedulerException::SchedulerException(const std::string& msg)
    : msg(msg)
{
}

SchedulerException::SchedulerException(const SchedulerException& copy)
    : msg(copy.msg)
{
}

std::string SchedulerException::message() const
{
    return msg;
}

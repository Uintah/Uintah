
#ifndef UINTAH_HOMEBREW_SchedulerException_H
#define UINTAH_HOMEBREW_SchedulerException_H

#include <SCICore/Exceptions/Exception.h>
#include <string>

class SchedulerException {
    std::string msg;
public:
    SchedulerException(const std::string&);
    virtual std::string message() const;
};

#endif

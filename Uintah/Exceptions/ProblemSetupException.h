
#ifndef UINTAH_HOMEBREW_ProblemSetupException_H
#define UINTAH_HOMEBREW_ProblemSetupException_H

#include <SCICore/Exceptions/Exception.h>
#include <string>

class ProblemSetupException : public SCICore::Exceptions::Exception {
    std::string msg;
public:
    ProblemSetupException(const std::string&);
    virtual const char* message() const;
};

#endif

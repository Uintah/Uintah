
#ifndef UINTAH_HOMEBREW_InvalidValue_H
#define UINTAH_HOMEBREW_InvalidValue_H

#include <SCICore/Exceptions/Exception.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <string>

class InvalidValue {
    std::string msg;
public:
    InvalidValue(const std::string&);
    InvalidValue(const std::string&, const Uintah::Interface::ProblemSpecP& context);
    InvalidValue(const InvalidValue&);
    virtual std::string message() const;

private:
    InvalidValue& operator=(const InvalidValue&);
};

#endif


#ifndef UINTAH_HOMEBREW_InvalidValue_H
#define UINTAH_HOMEBREW_InvalidValue_H

#include <SCICore/Exceptions/Exception.h>
#include <Uintah/Grid/ProblemSpecP.h>
#include <string>

class InvalidValue {
    std::string msg;
public:
    InvalidValue(const std::string&);
    InvalidValue(const std::string&, const Uintah::Grid::ProblemSpecP& context);
    virtual std::string message() const;
};

#endif


#ifndef UINTAH_HOMEBREW_ParticleException_H
#define UINTAH_HOMEBREW_ParticleException_H

#include <SCICore/Exceptions/Exception.h>
#include <string>

class ParticleException : public SCICore::Exceptions::Exception {
    std::string msg;
public:
    ParticleException(const std::string&);
    virtual std::string message() const;
};

#endif

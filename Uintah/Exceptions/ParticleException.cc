
#include "ParticleException.h"

ParticleException::ParticleException(const std::string& msg)
    : msg(msg)
{
}

const char* ParticleException::message() const
{
    return msg.c_str();
}


#include "ParticleException.h"

ParticleException::ParticleException(const std::string& msg)
    : msg(msg)
{
}

ParticleException::ParticleException(const ParticleException& copy)
    : msg(copy.msg)
{
}

const char* ParticleException::message() const
{
    return msg.c_str();
}

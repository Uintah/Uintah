
#include "ParticleException.h"

ParticleException::ParticleException(const std::string& msg)
    : msg(msg)
{
}

std::string ParticleException::message() const
{
    return msg;
}


#include <Packages/Uintah/Core/Grid/ParticleSet.h>

using namespace Uintah;

ParticleSet::ParticleSet()
    : d_numParticles(0)
{
}

ParticleSet::ParticleSet(particleIndex numParticles)
    : d_numParticles(numParticles)
{
}

ParticleSet::~ParticleSet()
{
}


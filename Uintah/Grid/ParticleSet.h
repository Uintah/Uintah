
#ifndef ParticleSet_h
#define ParticleSet_h

#include "RefCounted.h"
#include <iostream> // TEMPORARY

class ParticleSet : public RefCounted {
public:
    ParticleSet();
    ~ParticleSet();
    typedef int index;

    index addParticle() {
	return d_numParticles++;
    }
    index numParticles() {
	return d_numParticles;
    }
private:
    index d_numParticles;
    ParticleSet(const ParticleSet&);
    ParticleSet& operator=(const ParticleSet&);
};

#endif


#ifndef ParticleSubset_h
#define ParticleSubset_h

#include "ParticleSet.h"
#include "RefCounted.h"
#include <vector>

class ParticleSubset : public RefCounted {
public:
    ParticleSubset(ParticleSet* pset);
    ~ParticleSubset();

    ParticleSet* getParticleSet() const {
	return pset;
    }

    ParticleSet::index addParticle() {
	ParticleSet::index idx = pset->addParticle();
	particles.push_back(idx);
	return idx;
    }

    typedef std::vector<ParticleSet::index>::iterator iterator;
    iterator begin() {
	return particles.begin();
    }
    iterator end() {
	return particles.end();
    }

    ParticleSet::index numParticles() {
	return (ParticleSet::index)particles.size();
    }
    iterator seek(int idx) {
	return particles.begin()+idx;
    }
    void resize(ParticleSet::index newSize) {
	particles.resize(newSize);
    }

    void set(ParticleSet::index idx, ParticleSet::index value) {
	particles[idx] = value;
    }

private:
    ParticleSet* pset;
    std::vector<ParticleSet::index> particles;
    ParticleSubset();
    ParticleSubset(const ParticleSubset& copy);
    ParticleSubset& operator=(const ParticleSubset&);
};

#endif

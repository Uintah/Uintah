#ifndef PARTICLESUBSET_H
#define PARTICLESUBSET_H

#include "ParticleSet.h"
#include "RefCounted.h"
#include <vector>

namespace Uintah {
namespace Grid {

/**************************************

CLASS
   ParticleSubset
   
   Short description...

GENERAL INFORMATION

   ParticleSubset.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Particle, ParticleSet, ParticleSubset

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class ParticleSubset : public RefCounted {
public:
    ParticleSubset(ParticleSet* pset);
    ~ParticleSubset();

    //////////
    // Insert Documentation Here:
    ParticleSet* getParticleSet() const {
	return d_pset;
    }

    //////////
    // Insert Documentation Here:
    particleIndex addParticle() {
	particleIndex idx = d_pset->addParticle();
	d_particles.push_back(idx);
	return idx;
    }

    typedef std::vector<particleIndex>::iterator iterator;

    //////////
    // Insert Documentation Here:
    iterator begin() {
	return d_particles.begin();
    }

    //////////
    // Insert Documentation Here:
    iterator end() {
	return d_particles.end();
    }

    //////////
    // Insert Documentation Here:
    particleIndex numParticles() {
	return (particleIndex) d_particles.size();
    }

    //////////
    // Insert Documentation Here:
    iterator seek(int idx) {
	return d_particles.begin() + idx;
    }

    //////////
    // Insert Documentation Here:
    void resize(particleIndex newSize) {
	d_particles.resize(newSize);
    }

    //////////
    // Insert Documentation Here:
    void set(particleIndex idx, particleIndex value) {
	d_particles[idx] = value;
    }

private:
    //////////
    // Insert Documentation Here:
    ParticleSet*               d_pset;
    std::vector<particleIndex> d_particles;

    ParticleSubset();
    ParticleSubset(const ParticleSubset& copy);
    ParticleSubset& operator=(const ParticleSubset&);
};

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif

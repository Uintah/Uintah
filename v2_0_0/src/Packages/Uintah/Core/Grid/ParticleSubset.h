#ifndef PARTICLESUBSET_H
#define PARTICLESUBSET_H

#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class Patch;
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

  class ParticleVariableBase;

  class ParticleSubset : public RefCounted {
  public:
    ParticleSubset(ParticleSet* pset, bool fill,
		   int matlIndex, const Patch*,
		   particleIndex sizeHint);
    ParticleSubset(ParticleSet* pset, bool fill,
		   int matlIndex, const Patch*,
		   Ghost::GhostType gtype, int numGhostCells,
		   const std::vector<const Patch*>& neighbors,
		   const std::vector<ParticleSubset*>& subsets);
    ParticleSubset();
    ~ParticleSubset();
    
    //////////
    // Insert Documentation Here:
    ParticleSet* getParticleSet() const {
      return d_pset;
    }
      
    //////////
    // Insert Documentation Here:
    void addParticle(particleIndex idx) {
      if(d_numParticles >= d_allocatedSize)
	expand(1);
      d_particles[d_numParticles++] = idx;
    }
    particleIndex addParticles(particleIndex count);

    void resize(particleIndex idx);

    typedef particleIndex* iterator;
      
    //////////
    // Insert Documentation Here:
    iterator begin() {
      return d_particles;
    }
      
    //////////
    // Insert Documentation Here:

    iterator end() {
      return d_particles+d_numParticles;
    }
      
    //////////
    // Insert Documentation Here:
    const particleIndex* getPointer() const
    {
      return d_particles;
    }
      
    particleIndex* getPointer()
    {
      return d_particles;
    }
      
    //////////
    // Insert Documentation Here:
    particleIndex numParticles() {
      return d_numParticles;
    }
      
    //////////
    // Insert Documentation Here:
    void set(particleIndex idx, particleIndex value) {
      d_particles[idx] = value;
    }

    int numGhostCells() const {
      return d_numGhostCells;
    }
    const Patch* getPatch() const {
      return d_patch;
    }
    Ghost::GhostType getGhostType() const {
      return d_gtype;
    }
    int getMatlIndex() const {
      return d_matlIndex;
    }

    // sort the set by particle IDs
    void sort(ParticleVariableBase* particleIDs);

    const std::vector<const Patch*>& getNeighbors() const {
      return neighbors;
    }
    const std::vector<ParticleSubset*>& getNeighborSubsets() const {
      return neighbor_subsets;
    }
   private:
    //////////
    // Insert Documentation Here:
     
    ParticleSet*               d_pset;
    particleIndex* d_particles;
    particleIndex d_numParticles;
    particleIndex d_allocatedSize;
    int d_numExpansions;

    int d_matlIndex;
    const Patch* d_patch;
    Ghost::GhostType d_gtype;
    int d_numGhostCells;

    std::vector<const Patch*> neighbors;
    std::vector<ParticleSubset*> neighbor_subsets;

    void fillset();

    void init();
    void expand(particleIndex minSizeIncrement);

    ParticleSubset(const ParticleSubset& copy);
    ParticleSubset& operator=(const ParticleSubset&);
  };
} // End namespace Uintah

#endif

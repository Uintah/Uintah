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
		     int matlIndex, const Patch*);
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
      particleIndex addParticle() {
	 particleIndex idx = d_pset->addParticle();
	 d_particles.push_back(idx);
	 return idx;
      }
      
      //////////
      // Insert Documentation Here:
      void addParticle(particleIndex idx) {
	 d_particles.push_back(idx);
      }

      //////////
      // Insert Documentation Here:
      particleIndex addParticles(particleIndex count) {
	particleIndex oldsize = d_pset->addParticles(count);
	particleIndex newsize = oldsize+count;
	particleIndex start = (particleIndex)d_particles.size();
	d_particles.resize(d_particles.size()+newsize);
	for(particleIndex idx = oldsize; idx < newsize; idx++, start++)
	  d_particles[start] = idx;
	return oldsize;  // The beginning of the new index range
      }
      
      typedef std::vector<particleIndex>::iterator iterator;
      
      //////////
      // Insert Documentation Here:
      iterator begin() {
	 return d_particles.begin();
      }
      
      //////////
      // Insert Documentation Here:
      const particleIndex* getPointer() const
      {
       return &d_particles[0];
      }
      
      particleIndex* getPointer()
      {
       return &d_particles[0];
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
      void resize(particleIndex newSize) {
	 d_particles.resize(newSize);
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
      std::vector<particleIndex> d_particles;

      int d_matlIndex;
      const Patch* d_patch;
      Ghost::GhostType d_gtype;
      int d_numGhostCells;

      std::vector<const Patch*> neighbors;
      std::vector<ParticleSubset*> neighbor_subsets;

      void fillset();
      
      ParticleSubset(const ParticleSubset& copy);
      ParticleSubset& operator=(const ParticleSubset&);
   };

} // End namespace Uintah

#endif

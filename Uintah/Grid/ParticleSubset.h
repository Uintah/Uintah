#ifndef PARTICLESUBSET_H
#define PARTICLESUBSET_H

#include "ParticleSet.h"
#include "RefCounted.h"
#include <vector>

namespace Uintah {

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
      ParticleSubset(ParticleSet* pset, bool fill);
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
      
      ParticleSubset(const ParticleSubset& copy);
      ParticleSubset& operator=(const ParticleSubset&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/05/30 20:19:31  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.6  2000/05/20 08:09:24  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.5  2000/05/20 02:36:06  kuzimmer
// Multiple changes for new vis tools and DataArchive
//
// Revision 1.4  2000/05/10 20:03:01  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.3  2000/04/26 06:48:51  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif

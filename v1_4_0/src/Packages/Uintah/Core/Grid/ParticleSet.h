#ifndef PARTICLESET_H
#define PARTICLESET_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>

namespace Uintah {
   typedef int particleIndex;
   typedef int particleId;
   class ParticleSubset;

/**************************************

CLASS
   ParticleSet
   
   Short description...

GENERAL INFORMATION

   ParticleSet.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Particle, ParticleSet

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ParticleSet : public RefCounted {
   public:
      ParticleSet();
      ParticleSet(particleIndex numParticles);
      ~ParticleSet();
      
      //////////
      // Insert Documentation Here:
      particleIndex addParticle() {
	 return d_numParticles++;
      }
      
      //////////
      // Insert Documentation Here:
      particleIndex addParticles(particleIndex n) {
	particleIndex old_numParticles = d_numParticles;
	d_numParticles+=n;
	return old_numParticles;
      }
      
      //////////
      // Insert Documentation Here:
      particleIndex numParticles() {
	 return d_numParticles;
      }

   private:
      //////////
      // Insert Documentation Here:
      particleIndex d_numParticles;
      
      ParticleSet(const ParticleSet&);
      ParticleSet& operator=(const ParticleSet&);
   };
} // End namespace Uintah
   
#endif

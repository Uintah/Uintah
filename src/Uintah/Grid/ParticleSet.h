#ifndef PARTICLESET_H
#define PARTICLESET_H

#include "RefCounted.h"
#include <iostream> // TEMPORARY

namespace Uintah {

   typedef int particleIndex;

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
      void addParticles(particleIndex n) {
	  d_numParticles+=n;
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
   
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/05/10 20:03:01  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.4  2000/04/28 07:35:36  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.3  2000/04/26 06:48:51  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif

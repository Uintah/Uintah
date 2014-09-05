#ifndef __SHELL_PARTICLE_CREATOR_H__
#define __SHELL_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

/*******************************************************************************

CLASS

   ShellParticleCreator
   
   Creates particles that have properties of shells


GENERAL INFORMATION

   ShellParticleCreator.h

   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2003 University of Utah

KEYWORDS
   Shell particles

DESCRIPTION
   
   Same as membrane particle creator except that a thickness parameter
   is attached to shells along with the corotational tangents to the
   shell at the particle.  The directions of the directors of the shell 
   are calculated using the tangents.

WARNING
  
*******************************************************************************/
  class ShellParticleCreator : public ParticleCreator {
  public:
    
    /////////////////////////////////////////////////////////////////////////
    //
    // Constructor and destructor
    //
    ShellParticleCreator(MPMMaterial* matl, 
			 MPMLabel* lb,
			 int n8or27,
			 bool haveLoadCurve,
			 bool doErosion);
    virtual ~ShellParticleCreator();
    
    /////////////////////////////////////////////////////////////////////////
    //
    // Actually create particles using geometry
    //
    virtual ParticleSubset* createParticles(MPMMaterial* matl, 
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*, 
					    DataWarehouse* new_dw,
					    MPMLabel* lb,
					    vector<GeometryObject*>&);

    /////////////////////////////////////////////////////////////////////////
    //
    // Count particles in a patch
    //
    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&) ;
    virtual particleIndex countAndCreateParticles(const Patch*,
						  GeometryObject* obj) ;

    /////////////////////////////////////////////////////////////////////////
    //
    // Make sure data is copied when particles cross patch boundaries
    //
    virtual void registerPermanentParticleState(MPMMaterial* matl,
						MPMLabel* lb);

  };

} // End of namespace Uintah

#endif // __SHELL_PARTICLE_CREATOR_H__

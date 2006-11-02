#ifndef __IMPLICIT_PARTICLE_CREATOR_H__
#define __IMPLICIT_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class ImplicitParticleCreator : public ParticleCreator {
  public:
    
    ImplicitParticleCreator(MPMMaterial* matl, MPMFlags* flags);
    virtual ~ImplicitParticleCreator();

    virtual void registerPermanentParticleState(MPMMaterial* matl);

    virtual void initializeParticle(const Patch* patch,
				    vector<GeometryObject*>::const_iterator obj, 
				    MPMMaterial* matl,
				    Point p, IntVector cell_idx,
				    particleIndex i,
				    CCVariable<short int>& cellNAPI);

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
					      int dwi, const Patch* patch,
					      DataWarehouse* new_dw);
    

 
  protected:
    ParticleVariable<Vector> pacceleration;
    ParticleVariable<double> pvolumeold;
    ParticleVariable<double> pExternalHeatFlux;
    

 
  };



} // End of namespace Uintah

#endif // __IMPLICIT_PARTICLE_CREATOR_H__

#ifndef __FRACTURE_PARTICLE_CREATOR_H__
#define __FRACTURE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class FractureParticleCreator : public ParticleCreator {
  public:
    
    FractureParticleCreator(MPMMaterial* matl, MPMFlags* flags);
    virtual ~FractureParticleCreator();

    virtual void registerPermanentParticleState(MPMMaterial* matl);

    virtual void applyForceBC(const Vector& dxpp, const Point& pp,
                              const double& pMass, Vector& pExtForce);
    

    
  };



} // End of namespace Uintah

#endif // __FRACTURE_PARTICLE_CREATOR_H__

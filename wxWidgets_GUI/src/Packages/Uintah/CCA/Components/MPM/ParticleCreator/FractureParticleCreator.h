#ifndef __FRACTURE_PARTICLE_CREATOR_H__
#define __FRACTURE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class FractureParticleCreator : public ParticleCreator {
  public:
    
    FractureParticleCreator(MPMMaterial* matl, MPMFlags* flags);
    virtual ~FractureParticleCreator();

    virtual ParticleSubset* createParticles(MPMMaterial* matl,
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*, 
					    DataWarehouse* new_dw,
					    MPMLabel* lb, 
					    vector<GeometryObject*>&);

    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&) ;

    virtual particleIndex countAndCreateParticles(const Patch*,
						  GeometryObject* obj) ;

    virtual void registerPermanentParticleState(MPMMaterial* matl);

    void applyForceBC(const Vector& dxpp, 
                      const Point& pp,
                      const double& pMass, 
		      Vector& pExtForce);
    

    
  };



} // End of namespace Uintah

#endif // __FRACTURE_PARTICLE_CREATOR_H__

#ifndef __MEMBRANE_PARTICLE_CREATOR_H__
#define __MEMBRANE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class MembraneParticleCreator : public ParticleCreator {
  public:
    
    MembraneParticleCreator(MPMMaterial* matl, MPMLabel* lb,int n8or27);
    virtual ~MembraneParticleCreator();
    
    virtual ParticleSubset* createParticles(MPMMaterial* matl, 
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*, 
					    DataWarehouse* new_dw,
					    MPMLabel* lb,
					    vector<GeometryObject*>&);

    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&) const;
    virtual particleIndex countParticles(GeometryObject* obj,
					 const Patch*) const;

    virtual void registerPermanentParticleState(MPMMaterial* matl,
						MPMLabel* lb);

        
  };



} // End of namespace Uintah

#endif // __MEMBRANE_PARTICLE_CREATOR_H__

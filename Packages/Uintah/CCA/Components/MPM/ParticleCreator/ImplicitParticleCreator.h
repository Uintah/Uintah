#ifndef __IMPLICIT_PARTICLE_CREATOR_H__
#define __IMPLICIT_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class ImplicitParticleCreator : public ParticleCreator {
  public:
    
    ImplicitParticleCreator(MPMMaterial* matl, 
                           MPMLabel* lb,
                           int n8or27,
                           bool haveLoadCurve,
			   bool doErosion);
    virtual ~ImplicitParticleCreator();

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

#endif // __IMPLICIT_PARTICLE_CREATOR_H__

#ifndef __DEFAULT_PARTICLE_CREATOR_H__
#define __DEFAULT_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class DefaultParticleCreator : public ParticleCreator {
  public:
    
    DefaultParticleCreator(MPMMaterial* matl, 
                           MPMLabel* lb,
                           int n8or27,
                           bool haveLoadCurve,
			   bool doErosion);
    virtual ~DefaultParticleCreator();

    virtual ParticleSubset* createParticles(MPMMaterial* matl,
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*, 
					    DataWarehouse* new_dw,
					    MPMLabel* lb, 
					    vector<GeometryObject*>&);

    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&);
    virtual particleIndex countAndCreateParticles(const Patch*,
						  GeometryObject* obj);
    
    
  };



} // End of namespace Uintah

#endif // __DEFAULT_PARTICLE_CREATOR_H__

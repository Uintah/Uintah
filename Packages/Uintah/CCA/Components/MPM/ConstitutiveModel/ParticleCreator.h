#ifndef __PARTICLE_CREATOR_H__
#define __PARTICLE_CREATOR_H__

#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <vector>

namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;

  class GeometryObject;
  class Patch;
  class DataWarehouse;
  class MPMLabel;
  class MPMMaterial;

  class ParticleCreator {
  public:
    
    ParticleCreator();
    virtual ~ParticleCreator();

    virtual void createParticles(MPMMaterial* matl,particleIndex numParticles,
				 CCVariable<short int>& cellNAPID,
				 const Patch*,DataWarehouse* new_dw,
				 MPMLabel* lb,std::vector<GeometryObject*>&);

    particleIndex countParticles(const Patch*,
				 std::vector<GeometryObject*>&) const;
    particleIndex countParticles(GeometryObject* obj,
				 const Patch*) const;
    
  };



} // End of namespace Uintah

#endif // __PARTICLE_CREATOR_H__

#ifndef __MEMBRANE_PARTICLE_CREATOR_H__
#define __MEMBRANE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class MembraneParticleCreator : public ParticleCreator {
  public:
    
    MembraneParticleCreator();
    virtual ~MembraneParticleCreator();
    
    virtual void createParticles(MPMMaterial* matl, particleIndex numParticles,
				 CCVariable<short int>& cellNAPID,
				 const Patch*, DataWarehouse* new_dw,
				 MPMLabel* lb, std::vector<GeometryObject*>&);

    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&) const;
    virtual particleIndex countParticles(GeometryObject* obj,
					 const Patch*) const;
    
        
  };



} // End of namespace Uintah

#endif // __MEMBRANE_PARTICLE_CREATOR_H__

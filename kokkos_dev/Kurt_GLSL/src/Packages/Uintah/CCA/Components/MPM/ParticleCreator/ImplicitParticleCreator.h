#ifndef __IMPLICIT_PARTICLE_CREATOR_H__
#define __IMPLICIT_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class ImplicitParticleCreator : public ParticleCreator {
  public:
    
    ImplicitParticleCreator(MPMMaterial* matl, MPMFlags* flags);
    virtual ~ImplicitParticleCreator();

    virtual ParticleSubset* createParticles(MPMMaterial* matl, 
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*, 
					    DataWarehouse* new_dw,
					    vector<GeometryObject*>&);

    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&) ;
    virtual particleIndex countAndCreateParticles(const Patch*,
						  GeometryObject* obj) ;
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
    

 
  };



} // End of namespace Uintah

#endif // __IMPLICIT_PARTICLE_CREATOR_H__

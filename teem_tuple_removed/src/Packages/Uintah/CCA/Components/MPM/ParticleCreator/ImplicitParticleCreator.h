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
					 std::vector<GeometryObject*>&) ;
    virtual particleIndex countAndCreateParticles(const Patch*,
						  GeometryObject* obj) ;
    virtual void registerPermanentParticleState(MPMMaterial* matl,
						MPMLabel* lb);
 
  protected:

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
					      int dwi, MPMLabel* lb, 
					      const Patch* patch,
					      DataWarehouse* new_dw);
    
    virtual void initializeParticle(const Patch* patch,
				    vector<GeometryObject*>::const_iterator obj, 
				    MPMMaterial* matl,
				    Point p, IntVector cell_idx,
				    particleIndex i,
				    CCVariable<short int>& cellNAPI);


    ParticleVariable<Vector> pacceleration;
    ParticleVariable<double> pvolumeold;
    

 
  };



} // End of namespace Uintah

#endif // __IMPLICIT_PARTICLE_CREATOR_H__

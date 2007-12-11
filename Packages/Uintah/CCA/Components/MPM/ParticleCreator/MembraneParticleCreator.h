#ifndef __MEMBRANE_PARTICLE_CREATOR_H__
#define __MEMBRANE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class MembraneParticleCreator : public ParticleCreator {
  public:
    
    MembraneParticleCreator(MPMMaterial* matl,MPMFlags* flags);
    virtual ~MembraneParticleCreator();

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

    const VarLabel* pTang1Label;
    const VarLabel* pTang2Label;
    const VarLabel* pNormLabel;
    const VarLabel* pTang1Label_preReloc;
    const VarLabel* pTang2Label_preReloc;
    const VarLabel* pNormLabel_preReloc;


  protected:
    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
					      int dwi, const Patch* patch,
					      DataWarehouse* new_dw);
    

    ParticleVariable<Vector> pTang1, pTang2, pNorm;


        
  };



} // End of namespace Uintah

#endif // __MEMBRANE_PARTICLE_CREATOR_H__

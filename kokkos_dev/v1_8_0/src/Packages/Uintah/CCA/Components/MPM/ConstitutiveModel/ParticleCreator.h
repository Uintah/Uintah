#ifndef __PARTICLE_CREATOR_H__
#define __PARTICLE_CREATOR_H__

#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <vector>

namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;

  class GeometryObject;
  class Patch;
  class DataWarehouse;
  class MPMLabel;
  class MPMMaterial;
  class ParticleSubset;
  class VarLabel;

  class ParticleCreator {
  public:
    
    ParticleCreator(MPMMaterial* matl, MPMLabel* lb,int n8or27);
    virtual ~ParticleCreator();

    virtual ParticleSubset* createParticles(MPMMaterial* matl,
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*,DataWarehouse* new_dw,
					    MPMLabel* lb,
					    vector<GeometryObject*>&);

    virtual void registerPermanentParticleState(MPMMaterial* matl,
						MPMLabel* lb);

    virtual particleIndex countParticles(const Patch*,
				 std::vector<GeometryObject*>&) const;
    virtual particleIndex countParticles(GeometryObject* obj,
				 const Patch*) const;

    
  protected:
    
    void applyForceBC(const Vector& dxpp, 
                      const Point& pp,
                      const double& pMass, 
		      Vector& pExtForce);
    
    ParticleSubset* allocateVariables(particleIndex numParticles,
				      int dwi, MPMLabel* lb, 
				      const Patch* patch,
				      DataWarehouse* new_dw);

    ParticleVariable<Point> position;
    ParticleVariable<Vector> pvelocity, pexternalforce, psize;
    ParticleVariable<double> pmass, pvolume, ptemperature;
    ParticleVariable<long64> pparticleID;

    int d_8or27;

    vector<const VarLabel* > particle_state, particle_state_preReloc;
  };



} // End of namespace Uintah

#endif // __PARTICLE_CREATOR_H__

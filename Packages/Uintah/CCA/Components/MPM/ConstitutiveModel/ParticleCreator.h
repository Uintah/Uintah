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

  class ParticleCreator {
  public:
    
    ParticleCreator();
    virtual ~ParticleCreator();

    virtual ParticleSubset* createParticles(MPMMaterial* matl,
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*,DataWarehouse* new_dw,
					    MPMLabel* lb,
					    vector<GeometryObject*>&);

    particleIndex countParticles(const Patch*,
				 std::vector<GeometryObject*>&) const;
    particleIndex countParticles(GeometryObject* obj,
				 const Patch*) const;

  protected:
    
    void applyForceBC(particleIndex start, 
		      ParticleVariable<Vector>& pextforce,
		      ParticleVariable<double>& pmass, 
		      ParticleVariable<Point>& position);
    
    ParticleSubset* allocateVariables(particleIndex numParticles,
					      int dwi, MPMLabel* lb, 
					      const Patch* patch,
					      DataWarehouse* new_dw);

    ParticleVariable<Point> position;
    ParticleVariable<Vector> pvelocity, pexternalforce, psize;
    ParticleVariable<double> pmass, pvolume, ptemperature;
    ParticleVariable<long64> pparticleID;

    
  };



} // End of namespace Uintah

#endif // __PARTICLE_CREATOR_H__

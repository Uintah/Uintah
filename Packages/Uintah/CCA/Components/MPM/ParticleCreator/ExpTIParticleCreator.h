#ifndef __EXPTI_PARTICLE_CREATOR_H__
#define __EXPTI_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

namespace Uintah {

  class ExpTIParticleCreator : public ParticleCreator {
  public:
    
    ExpTIParticleCreator(MPMMaterial* matl, 
                           MPMLabel* lb,
                           int n8or27,
                           bool haveLoadCurve,
			   bool doErosion);
    virtual ~ExpTIParticleCreator();

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


    ParticleVariable<Vector> pfiberdir;
    vector<const VarLabel* > particle_state, particle_state_preReloc;
//    typedef map<pair<const Patch*,GeometryObject*>,vector<Point> > geompoints;
//    geompoints d_object_points;
//    typedef map<pair<const Patch*,GeometryObject*>,vector<double> > geomvols;
//    geomvols d_object_vols;
//    typedef map<pair<const Patch*,GeometryObject*>,vector<Vector> > geomvecs;
//    geomvecs d_object_forces;
    geomvecs d_object_fibers;
  };



} // End of namespace Uintah

#endif // __EXPTI_PARTICLE_CREATOR_H__

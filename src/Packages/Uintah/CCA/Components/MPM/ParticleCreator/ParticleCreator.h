#ifndef __PARTICLE_CREATOR_H__
#define __PARTICLE_CREATOR_H__

#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;

  class GeometryObject;
  class GeometryPiece;
  class Patch;
  class DataWarehouse;
  class MPMLabel;
  class MPMMaterial;
  class ParticleSubset;
  class VarLabel;

  class ParticleCreator {
  public:
    
    ParticleCreator(MPMMaterial* matl, 
                    MPMLabel* lb,
                    int n8or27, bool haveLoadCurve, bool doErosion);

    virtual ~ParticleCreator();

    virtual ParticleSubset* createParticles(MPMMaterial* matl,
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*,DataWarehouse* new_dw,
					    MPMLabel* lb,
					    vector<GeometryObject*>&);

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
					      int dwi, MPMLabel* lb, 
					      const Patch* patch,
					      DataWarehouse* new_dw);

    virtual void allocateVariablesAddRequires(Task* task, 
					      const MPMMaterial* matl,
					      const PatchSet* patch, 
					      MPMLabel* lb) const;

    virtual void allocateVariablesAdd(MPMLabel* lb, DataWarehouse* new_dw,
				      ParticleSubset* addset,
				      map<const VarLabel*,ParticleVariableBase*>* newState,
				      ParticleSubset* delset,
				      DataWarehouse* old_dw);

    virtual void registerPermanentParticleState(MPMMaterial* matl,
						MPMLabel* lb);

    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&);

    virtual particleIndex countAndCreateParticles(const Patch*,
						  GeometryObject* obj);

    virtual vector<const VarLabel* > returnParticleState();

  protected:

    void createPoints(const Patch* patch, GeometryObject* obj);

    virtual void initializeParticle(const Patch* patch,
				    vector<GeometryObject*>::const_iterator obj, 
				    MPMMaterial* matl,
				    Point p, IntVector cell_idx,
				    particleIndex i,
				    CCVariable<short int>& cellNAPI);
    
    // Get the LoadCurveID applicable for this material point
    int getLoadCurveID(const Point& pp, const Vector& dxpp);

    // Print MPM physical boundary condition information
    void printPhysicalBCs();

    void applyForceBC(const Vector& dxpp, 
                      const Point& pp,
                      const double& pMass, 
		      Vector& pExtForce);
    
    int checkForSurface(const GeometryPiece* piece, const Point p,
                        const Vector dxpp);
    

    ParticleVariable<Point> position;
    ParticleVariable<Vector> pvelocity, pexternalforce, psize;
    ParticleVariable<double> pmass, pvolume, ptemperature, psp_vol,perosion;
    ParticleVariable<long64> pparticleID;
    ParticleVariable<Vector> pdisp;

    ParticleVariable<int> pLoadCurveID;

    int d_8or27;
    bool d_useLoadCurves;
    bool d_doErosion;

    vector<const VarLabel* > particle_state, particle_state_preReloc;
    typedef map<pair<const Patch*,GeometryObject*>,vector<Point> > geompoints;
    geompoints d_object_points;
  };



} // End of namespace Uintah

#endif // __PARTICLE_CREATOR_H__

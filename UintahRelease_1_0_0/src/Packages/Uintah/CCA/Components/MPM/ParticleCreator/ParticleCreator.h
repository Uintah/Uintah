#ifndef __PARTICLE_CREATOR_H__
#define __PARTICLE_CREATOR_H__

#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;

  class GeometryObject;
  class Patch;
  class DataWarehouse;
  class MPMFlags;
  class MPMMaterial;
  class MPMLabel;
  class ParticleSubset;
  class VarLabel;

  class ParticleCreator {
  public:
    
    ParticleCreator(MPMMaterial* matl, MPMFlags* flags);


    virtual ~ParticleCreator();


    virtual ParticleSubset* createParticles(MPMMaterial* matl,
					    particleIndex numParticles,
					    CCVariable<short int>& cellNAPID,
					    const Patch*,DataWarehouse* new_dw,
                                            vector<GeometryObject*>&);


    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
					      int dwi, const Patch* patch,
					      DataWarehouse* new_dw);

    virtual void allocateVariablesAddRequires(Task* task, 
					      const MPMMaterial* matl,
					      const PatchSet* patch) const;

    virtual void allocateVariablesAdd(DataWarehouse* new_dw,
				      ParticleSubset* addset,
				      map<const VarLabel*,ParticleVariableBase*>* newState,
				      ParticleSubset* delset,
				      DataWarehouse* old_dw);

    virtual void registerPermanentParticleState(MPMMaterial* matl);

    virtual particleIndex countParticles(const Patch*,
					 std::vector<GeometryObject*>&);

    virtual particleIndex countAndCreateParticles(const Patch*,
						  GeometryObject* obj);

    vector<const VarLabel* > returnParticleState();
    vector<const VarLabel* > returnParticleStatePreReloc();

  protected:

    void createPoints(const Patch* patch, GeometryObject* obj);



    virtual void initializeParticle(const Patch* patch,
				    vector<GeometryObject*>::const_iterator obj,
				    MPMMaterial* matl,
				    Point p, IntVector cell_idx,
				    particleIndex i,
				    CCVariable<short int>& cellNAPI);
    
    //////////////////////////////////////////////////////////////////////////
    /*! Get the LoadCurveID applicable for this material point */
    //////////////////////////////////////////////////////////////////////////
    int getLoadCurveID(const Point& pp, const Vector& dxpp);

    //////////////////////////////////////////////////////////////////////////
    /*! Print MPM physical boundary condition information */
    //////////////////////////////////////////////////////////////////////////
    void printPhysicalBCs();

    //////////////////////////////////////////////////////////////////////////
    /*! Calculate the external force to be applied to a particle */
    //////////////////////////////////////////////////////////////////////////
    virtual void applyForceBC(const Vector& dxpp,  const Point& pp,
                              const double& pMass,  Vector& pExtForce);
    
    int checkForSurface(const GeometryPieceP piece, const Point p,
                        const Vector dxpp);
    

    ParticleVariable<Point> position;
    ParticleVariable<Vector> pvelocity, pexternalforce, psize;
    ParticleVariable<double> pmass, pvolume, ptemperature, psp_vol,perosion;
    ParticleVariable<long64> pparticleID;
    ParticleVariable<Vector> pdisp;
    ParticleVariable<Vector> pfiberdir;
    ParticleVariable<double> ptempPrevious;  // for thermal stress 

    ParticleVariable<int> pLoadCurveID;

    MPMLabel* d_lb;

    bool d_useLoadCurves;
    bool d_with_color;

    // for thermal stress 
    double d_ref_temp;  // Thermal stress of the system is zero at d_ref_temp

    vector<const VarLabel* > particle_state, particle_state_preReloc;
    typedef map<pair<const Patch*,GeometryObject*>,vector<Point> > geompoints;
    geompoints d_object_points;
    typedef map<pair<const Patch*,GeometryObject*>,vector<double> > geomvols;
    geomvols d_object_vols;
    geomvols d_object_temps;
    typedef map<pair<const Patch*,GeometryObject*>,vector<Vector> > geomvecs;
    geomvecs d_object_forces;
    geomvecs d_object_fibers;
  };



} // End of namespace Uintah

#endif // __PARTICLE_CREATOR_H__

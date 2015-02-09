/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __PARTICLE_CREATOR_H__
#define __PARTICLE_CREATOR_H__

#include <Core/Thread/CrowdMonitor.h>

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <vector>
#include <map>

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


    virtual particleIndex createParticles(MPMMaterial* matl,
                                            CCVariable<short int>& cellNAPID,
                                            const Patch*,DataWarehouse* new_dw,
                                            std::vector<GeometryObject*>&);



    virtual void registerPermanentParticleState(MPMMaterial* matl);

    std::vector<const VarLabel* > returnParticleState();
    std::vector<const VarLabel* > returnParticleStatePreReloc();

    
    typedef std::map<GeometryObject*,std::vector<Point> > geompoints;
    typedef std::map<GeometryObject*,std::vector<double> > geomvols;
    typedef std::map<GeometryObject*,std::vector<Vector> > geomvecs;
    typedef std::map<GeometryObject*,std::vector<Matrix3> > geomMat3s;
  
    typedef struct {
    geompoints d_object_points;
    geomvols d_object_vols;
    geomvols d_object_temps;
    geomvols d_object_colors;
    geomvecs d_object_forces;
    geomvecs d_object_fibers;  
    geomvecs d_object_velocity; // gcd add
    geomMat3s d_object_size;  
    } ObjectVars;

    typedef struct {
    ParticleVariable<Point> position;
    ParticleVariable<Vector> pvelocity, pexternalforce;
    ParticleVariable<Matrix3> psize,pvelGrad;
    ParticleVariable<double> pmass, pvolume, ptemperature, psp_vol,perosion;
    ParticleVariable<double> pcolor,ptempPrevious,p_q;
    ParticleVariable<long64> pparticleID;
    ParticleVariable<Vector> pdisp;
    ParticleVariable<Vector> pfiberdir; 
    ParticleVariable<int> pLoadCurveID;
    ParticleVariable<int> plocalized;
    ParticleVariable<int> prefined;
    // ImplicitParticleCreator
    ParticleVariable<Vector> pacceleration;
    ParticleVariable<double> pvolumeold;
    ParticleVariable<double> pExternalHeatFlux;
    //MembraneParticleCreator
    ParticleVariable<Vector> pTang1, pTang2, pNorm;
    } ParticleVars;

  protected:

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
                                              int dwi, const Patch* patch,
                                              DataWarehouse* new_dw,
                                              ParticleVars& pvars);

    virtual particleIndex countAndCreateParticles(const Patch*,
                                                  GeometryObject* obj,
                                                  ObjectVars& vars);

    void createPoints(const Patch* patch, GeometryObject* obj, ObjectVars& vars);



    virtual void initializeParticle(const Patch* patch,
                                    std::vector<GeometryObject*>::const_iterator obj,
                                    MPMMaterial* matl,
                                    Point p, IntVector cell_idx,
                                    particleIndex i,
                                    CCVariable<short int>& cellNAPI,
                                    ParticleVars& pvars);
    
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
    

    MPMLabel* d_lb;
    MPMFlags* d_flags;

    bool d_useLoadCurves;
    bool d_with_color;
    bool d_artificial_viscosity;
    bool d_computeScaleFactor;
    bool d_useCPTI;

    std::vector<const VarLabel* > particle_state, particle_state_preReloc;

    
    mutable CrowdMonitor   d_lock;
  };



} // End of namespace Uintah

#endif // __PARTICLE_CREATOR_H__

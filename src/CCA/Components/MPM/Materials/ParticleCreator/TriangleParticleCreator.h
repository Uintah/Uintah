/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#ifndef __TRIANGLE_PARTICLE_CREATOR_H__
#define __TRIANGLE_PARTICLE_CREATOR_H__

#include "ParticleCreator.h"

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
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

  class TriangleParticleCreator : public ParticleCreator {
  public:
    
    TriangleParticleCreator(MPMMaterial* matl, MPMFlags* flags);


    virtual ~TriangleParticleCreator();


    virtual particleIndex createParticles(MPMMaterial* matl,
                                          CCVariable<int>& cellNAPID,
                                          const Patch*,DataWarehouse* new_dw,
                                          std::vector<GeometryObject*>&);



    virtual void registerPermanentParticleState(MPMMaterial* matl);

    std::vector<const VarLabel* > returnParticleState();
    std::vector<const VarLabel* > returnParticleStatePreReloc();

    
    typedef struct {
    ParticleVariable<Point> position;
    ParticleVariable<Vector> pvelocity, pexternalforce;
    ParticleVariable<Matrix3> psize,pvelGrad;
    ParticleVariable<double> pmass, pvolume, ptemperature, psp_vol,perosion;
    ParticleVariable<double> pcolor,ptempPrevious,p_q;
    ParticleVariable<double> psurface;
    ParticleVariable<int> pmodalID;
    ParticleVariable<Vector> psurfgrad;
    ParticleVariable<long64> pparticleID;
    ParticleVariable<Vector> pdisp,pTempGrad,parea;
    ParticleVariable<Vector> pfiberdir; 
    ParticleVariable<IntVector> pLoadCurveID;
    ParticleVariable<int> plocalized;
    ParticleVariable<int> prefined;
    ParticleVariable<int> pLastLevel;
    // ImplicitParticleCreator
    ParticleVariable<Vector> pacceleration;
    ParticleVariable<double> pvolumeold;
    ParticleVariable<double> pExternalHeatFlux;
    //Scalar Diffusion
    ParticleVariable<double> pConcentration;
    ParticleVariable<double> pConcPrevious;
    ParticleVariable<Vector> pConcGrad;
    ParticleVariable<double> pExternalScalarFlux;
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
                                    CCVariable<int>& cellNAPI,
                                    ParticleVars& pvars);
    
    //////////////////////////////////////////////////////////////////////////
    /*! Get the LoadCurveID applicable for this material point */
    //////////////////////////////////////////////////////////////////////////
    IntVector getLoadCurveID(const Point& pp, const Vector& dxpp, 
                             Vector& areacomps, int dwi);

    //////////////////////////////////////////////////////////////////////////
    /*! Print MPM physical boundary condition information */
    //////////////////////////////////////////////////////////////////////////
    void printPhysicalBCs();

    //////////////////////////////////////////////////////////////////////////
    /*! Calculate the external force to be applied to a particle */
    //////////////////////////////////////////////////////////////////////////
    int checkForSurface(const GeometryPieceP piece, const Point p,
                        const Vector dxpp);

    double checkForSurface2(const GeometryPieceP piece, const Point p,
                            const Vector dxpp);

    MPMLabel* d_lb;
    MPMFlags* d_flags;

    bool d_useLoadCurves;
    bool d_with_color;
    bool d_artificial_viscosity;
    bool d_computeScaleFactor;

    std::vector<const VarLabel* > particle_state, particle_state_preReloc;
    
  };

} // End of namespace Uintah

#endif // __TRIANGLE_PARTICLE_CREATOR_H__

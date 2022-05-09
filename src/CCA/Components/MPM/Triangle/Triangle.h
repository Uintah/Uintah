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

#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
#include <CCA/Ports/Scheduler.h>
#include <vector>
#include <map>

namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;

  class GeometryObject;
  class Patch;
  class DataWarehouse;
  class MPMFlags;
  class TriangleMaterial;
  class TriangleLabel;
  class MPMLabel;
  class ParticleSubset;
  class VarLabel;

  class Triangle {
  public:
    
    Triangle(TriangleMaterial* tm, MPMFlags* flags, MaterialManagerP& ss);

    virtual ~Triangle();

    virtual ParticleSubset* createTriangles(TriangleMaterial* matl,
                                            particleIndex numParticles,
                                            const Patch*,DataWarehouse* new_dw,
                                            const std::string filename);

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
                                              int dwi, const Patch* patch,
                                              DataWarehouse* new_dw);

    virtual void registerPermanentTriangleState(TriangleMaterial* tm);

    virtual particleIndex countTriangles(const Patch*, 
                                         const std::string fname);

    void scheduleInitialize(const LevelP& level, SchedulerP& sched,
                            MaterialManagerP &ss);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    std::vector<const VarLabel* > returnTriangleState();
    std::vector<const VarLabel* > returnTriangleStatePreReloc();

  protected:

    ParticleVariable<Point>   triangle_pos;
    ParticleVariable<long64>  triangleID;
    ParticleVariable<Matrix3> triangleSize;
    ParticleVariable<Matrix3> triangleDefGrad;
    ParticleVariable<Vector>  triangleMidToNode0;
    ParticleVariable<Vector>  triangleMidToNode1;
    ParticleVariable<Vector>  triangleMidToNode2;
    ParticleVariable<Vector>  triangleAreaAtNodes;
    ParticleVariable<double>  triangleArea;
    ParticleVariable<double>  triangleClay;  // 0 - no clay, 1 - full clay
    ParticleVariable<Vector>  triangleNormal;
    ParticleVariable<double>  triangleMassDisp;
    ParticleVariable<IntVector>  triangleUseInPenalty;
    ParticleVariable<Matrix3>    triangleNearbyMats;

    TriangleLabel* d_Tl;
    MPMLabel* d_lb;
    MPMFlags* d_flags;
    MaterialManagerP d_materialManager;

    std::vector<const VarLabel* > d_triangle_state, d_triangle_state_preReloc;
  };

} // End of namespace Uintah

#endif // __TRIANGLE_H

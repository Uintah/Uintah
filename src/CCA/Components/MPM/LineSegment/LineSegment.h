/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

#ifndef __LINESEGMENT_H
#define __LINESEGMENT_H

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
  class LineSegmentMaterial;
  class MPMLabel;
  class LineSegmentLabel;
  class ParticleSubset;
  class VarLabel;

  class LineSegment {
  public:
    
    LineSegment(LineSegmentMaterial* tm, MPMFlags* flags, MaterialManagerP& ss);

    virtual ~LineSegment();

    virtual ParticleSubset* createLineSegments(LineSegmentMaterial* matl,
                                            particleIndex numParticles,
                                            const Patch*,DataWarehouse* new_dw,
                                            const std::string filename);

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
                                              int dwi, const Patch* patch,
                                              DataWarehouse* new_dw);

    virtual void registerPermanentLineSegmentState(LineSegmentMaterial* tm);

    virtual particleIndex countLineSegments(const Patch*, 
                                            const std::string fname);

    void scheduleInitialize(const LevelP& level, SchedulerP& sched,
                            MaterialManagerP &ss);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    std::vector<const VarLabel* > returnLineSegmentState();
    std::vector<const VarLabel* > returnLineSegmentStatePreReloc();

  protected:

    ParticleVariable<Point>   lineseg_pos;
    ParticleVariable<long64>  linesegID;
    ParticleVariable<Matrix3> linesegSize;
    ParticleVariable<Matrix3> linesegDefGrad;
    ParticleVariable<Vector>  linesegMidToEnd;

    MPMLabel* d_lb;
    LineSegmentLabel* d_Ll;
    MPMFlags* d_flags;
    MaterialManagerP d_materialManager;

    std::vector<const VarLabel* > d_lineseg_state, d_lineseg_state_preReloc;
  };

} // End of namespace Uintah

#endif // __LINESEGMENT_H

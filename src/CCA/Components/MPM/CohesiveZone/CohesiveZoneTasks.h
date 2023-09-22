/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef __COHESIVE_ZONE_TASKS_H_
#define __COHESIVE_ZONE_TASKS_H_

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
  class CZMaterial;
  class MPMLabel;
  class CZLabel;
  class ParticleSubset;
  class VarLabel;

  class CohesiveZoneTasks {
  public:
    
    CohesiveZoneTasks(MaterialManagerP& ss, MPMFlags* flags);

    virtual ~CohesiveZoneTasks();

    virtual void cohesiveZoneProblemSetup(const ProblemSpecP& prob_spec,
                                          MPMFlags* flags);

    virtual void scheduleAddCohesiveZoneForces(SchedulerP&,
                                               const PatchSet*,
                                               const MaterialSubset*,
                                               const MaterialSubset*,
                                               const MaterialSet*);

    virtual void addCohesiveZoneForces(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

    virtual void scheduleUpdateCohesiveZones(SchedulerP&, 
                                             const PatchSet*,
                                             const MaterialSubset*,
                                             const MaterialSubset*,
                                             const MaterialSet*);

    virtual void updateCohesiveZones(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState;
    std::vector<std::vector<const VarLabel* > > d_cohesiveZoneState_preReloc;

  protected:

    MPMLabel* d_lb;
    CZLabel*  d_Cl;
    MPMFlags* d_flags;
    MaterialManagerP d_materialManager;
    int NGN, NGP;
  };

} // End of namespace Uintah

#endif // __COHESIVE_ZONE_TASKS_H_

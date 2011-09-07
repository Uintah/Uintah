/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __COHESIVE_ZONE_H_
#define __COHESIVE_ZONE_H_

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
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
  class ParticleSubset;
  class VarLabel;

  class CohesiveZone {
  public:
    
    CohesiveZone(CZMaterial* czmat, MPMFlags* flags,  SimulationStateP& ss);

    virtual ~CohesiveZone();

    virtual ParticleSubset* createCohesiveZones(CZMaterial* matl,
                                            particleIndex numParticles,
                                            CCVariable<short int>& cellNAPID,
                                            const Patch*,DataWarehouse* new_dw,
                                            const string filename);

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
                                              int dwi, const Patch* patch,
                                              DataWarehouse* new_dw);

    virtual void registerPermanentCohesiveZoneState(CZMaterial* czmat);

    virtual particleIndex countCohesiveZones(const Patch*, const string fname);

    void scheduleInitialize(const LevelP& level, SchedulerP& sched,
                            CZMaterial* czmat);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    vector<const VarLabel* > returnCohesiveZoneState();
    vector<const VarLabel* > returnCohesiveZoneStatePreReloc();

  protected:

    ParticleVariable<Point> czposition;
    ParticleVariable<Vector> cznormal, cztang, czdisptop, czdispbottom;
    ParticleVariable<double> czlength;
    ParticleVariable<long64> czID;
    ParticleVariable<Vector> czSeparation, czForce;
    ParticleVariable<int> czTopMat, czBotMat;
    ParticleVariable<int> czFailed;

    MPMLabel* d_lb;
    MPMFlags* d_flags;
    SimulationStateP d_sharedState;

    vector<const VarLabel* > d_cz_state, d_cz_state_preReloc;
  };

} // End of namespace Uintah

#endif // __COHESIVE_ZONE_H_

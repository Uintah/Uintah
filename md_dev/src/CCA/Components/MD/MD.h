/*

 The MIT License

 Copyright (c) 1997-2012 The University of Utah

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

#ifndef UINTAH_MD_H
#define UINTAH_MD_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/SPMEGrid.h>
#include <CCA/Components/MD/SPMEGridMap.h>
#include <CCA/Components/MD/SPMEMapPoint.h>
#include <CCA/Components/MD/Transformation3D.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <cstdio>
#include <cstring>
#include <vector>
#include <complex>

namespace Uintah {

class SimpleMaterial;

class MD : public UintahParallelComponent, public SimulationInterface {

  public:
    MD(const ProcessorGroup* myworld);

    virtual ~MD();

    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP&);

    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);

    virtual void scheduleComputeStableTimestep(const LevelP& level,
                                               SchedulerP&);

    virtual void scheduleTimeAdvance(const LevelP& level,
                                     SchedulerP&);

    virtual void scheduleSPME();

  protected:

    void scheduleCalculateNonBondedForces(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls);

    void scheduleUpdatePosition(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls);

  private:

    inline bool containsAtom(const IntVector &l,
                             const IntVector &h,
                             const Point &p)
    {
      return ((p.x() >= l.x() && p.x() < h.x()) && (p.y() >= l.y() && p.y() < h.y()) && (p.z() >= l.z() && p.z() < h.z()));
    }

    void generateNeighborList();

    std::vector<Point> calcReducedCoords(const std::vector<Point>& localRealCoordinates,
                                         const Transformation3D<std::complex<double> >& invertSpace);

    void extractCoordinates();

    bool isNeighbor(const Point* atom1,
                    const Point* atom2);

    void initialize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    void computeStableTimestep(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);

    vector<double> generateMVector(int Points,
                                   int Shift,
                                   int Max);

    vector<std::complex<double> > generateBVector(const int& points,
                                                  const vector<double>& M,
                                                  const int& max,
                                                  const int& splineOrder,
                                                  const vector<double>& splineCoeff);

    void calculateStaticGrids(const SPMEGrid<std::complex<double> >& localSPMEGrid,
                              const ParticleSubset& system,
                              SPMEGrid<std::complex<double> >& fB,
                              SPMEGrid<std::complex<double> >& fC,
                              SPMEGrid<std::complex<double> >& stressPreMult);

    void solveSPMECharge();

    SPMEGridMap<double> createSPMEChargeMap(const SPMEGrid<std::complex<double> >& SPMEGlobalGrid,
                                            const Patch& CurrentPatch,
                                            const ParticleSubset& globalParticleSubset);

    void spmeMapChargeToGrid(SPMEGrid<std::complex<double> >& LocalGridCopy,
                             const SPMEGridMap<std::complex<double> >& LocalGridMap,
                             const Patch& CurrentPatch);

    SPMEGrid<double> fC(const IntVector& GridExtents,
                        const ParticleSubset& globalParticleSubset);

    SPMEGrid<std::complex<double> > fB(const IntVector& GridExtents,
                                       const ParticleSubset& globalParticleSubset);

    void calculateNonBondedForces(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

    void updatePosition(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw);

    MDLabel* lb;
    SimulationStateP d_sharedState_;
    SimpleMaterial* mymat_;
    double delt_;

    std::vector<std::vector<const VarLabel*> > d_particleState;
    std::vector<std::vector<const VarLabel*> > d_particleState_preReloc;

    // fields specific to non-bonded interaction (LJ Potential)
    string coordinateFile_;
    unsigned int numAtoms_;
    double cutoffRadius_;  // the short ranged cut off distances (in Angstroms)
    Vector box_;  // the size of simulation
    double R12_;  // this is the v.d.w. repulsive parameter
    double R6_;  // this is the v.d.w. attractive parameter

    // neighborList[i] contains the index of all atoms located within a short ranged cut off from atom "i"
    std::vector<Point> atomList;
    std::vector<vector<int> > neighborList;

    // copy constructor and operator=
    MD(const MD&);
    MD& operator=(const MD&);
};
}

#endif

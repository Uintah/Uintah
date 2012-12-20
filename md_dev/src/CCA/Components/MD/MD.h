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
#include <CCA/Components/MD/Electrostatics.h>
#include <CCA/Components/MD/SPMEGrid.h>
#include <CCA/Components/MD/SPMEGridMap.h>
#include <CCA/Components/MD/MapPoint.h>
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

    /**
     * @brief
     * @param
     */
    MD(const ProcessorGroup* myworld);

    /**
     * @brief
     * @param
     */
    virtual ~MD();

    /**
     * @brief
     * @param
     * @return
     */
    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP&);

    /**
     * @brief
     * @param
     * @return
     */
    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);

    /**
     * @brief
     * @param
     * @return
     */
    virtual void scheduleComputeStableTimestep(const LevelP& level,
                                               SchedulerP&);

    /**
     * @brief
     * @param
     * @return
     */
    virtual void scheduleTimeAdvance(const LevelP& level,
                                     SchedulerP&);

    /**
     * @brief
     * @param
     * @return
     */
    enum IntegratorType {
      Explicit,
      Implicit,
    };

  protected:

    /**
     * @brief
     * @param
     * @return
     */
    void scheduleSetGridBoundaryConditions(SchedulerP&,
                                           const PatchSet*,
                                           const MaterialSet* matls);

    /**
     * @brief
     * @param
     * @return
     */
    void scheduleCalculateNonBondedForces(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls);

    /**
     * @brief
     * @param
     * @return
     */
    void scheduleInterpolateParticlesToGrid(SchedulerP&,
                                            const PatchSet*,
                                            const MaterialSet*);

    /**
     * @brief
     * @param
     * @return
     */
    void schedulePerformSPME(SchedulerP& sched,
                             const PatchSet* patched,
                             const MaterialSet* matls);

    /**
     * @brief
     * @param
     * @return
     */
    void scheduleUpdatePosition(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls);

  private:

    /**
     * @brief
     * @param
     * @return
     */
    void initialize(const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return
     */
    void computeStableTimestep(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return
     */
    void interpolateParticlesToGrid(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return
     */
    void performSPME(const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return
     */
    void calculateNonBondedForces(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return
     */
    void updatePosition(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return
     */
    inline bool containsAtom(const IntVector &l,
                             const IntVector &h,
                             const Point &p) const
    {
      return ((p.x() >= l.x() && p.x() < h.x()) && (p.y() >= l.y() && p.y() < h.y()) && (p.z() >= l.z() && p.z() < h.z()));
    }

    /**
     * @brief
     * @param
     * @return
     */
    void generateNeighborList();

    /**
     * @brief
     * @param
     * @return
     */
    std::vector<Point> calcReducedCoords(const std::vector<Point>& localRealCoordinates,
                                         const Transformation3D<std::complex<double> >& invertSpace);

    /**
     * @brief
     * @param
     * @return
     */
    void extractCoordinates();

    /**
     * @brief
     * @param
     * @return
     */
    bool isNeighbor(const Point* atom1,
                    const Point* atom2);

    MDLabel* lb;                     //!<
    SimulationStateP d_sharedState_;  //!<
    SimpleMaterial* mymat_;          //!<
    IntegratorType d_integrator;     //!<
    double delt_;                    //!<

    std::vector<std::vector<const VarLabel*> > d_particleState;  //!<
    std::vector<std::vector<const VarLabel*> > d_particleState_preReloc;  //!<
//    vector<const VarLabel* > particle_state, particle_state_preReloc;

    // fields specific to non-bonded interaction (LJ Potential)
    string coordinateFile_;  //!<
    unsigned int numAtoms_;  //!<
    double cutoffRadius_;    //!< The short ranged cut off distances (in Angstroms)
    Vector box_;             //!< The size of simulation
    double R12_;             //!< This is the v.d.w. repulsive parameter
    double R6_;              //!< This is the v.d.w. attractive parameter

    // neighborList[i] contains the index of all atoms located within a short ranged cut off from atom "i"
    std::vector<Point> atomList;             //!<
    std::vector<vector<int> > neighborList;  //!<

    Electrostatics* elctrostatics;           //!<

    mutable CrowdMonitor   d_lock;           //!<

    // copy constructor and operator=
    MD(const MD&);                           //!<
    MD& operator=(const MD&);                //!<
};
}

#endif

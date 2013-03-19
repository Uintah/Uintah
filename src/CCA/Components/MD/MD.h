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
#include <CCA/Components/MD/SPME.h>
#include <CCA/Components/MD/SPMEMapPoint.h>
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

typedef int particleIndex;
typedef int particleId;

class SimpleMaterial;
class SPME;

/**
 *  @class MD
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   December, 2012
 *
 *  @brief
 *
 *  @param
 */
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
      Explicit, Implicit,
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
    void scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
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
    void registerPermanentParticleState(SimpleMaterial* matl);

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
    void interpolateToParticlesAndUpdate(const ProcessorGroup* pg,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    /**
     * @brief
     * @param
     * @return
     */
    inline bool containsAtom(const IntVector& l,
                             const IntVector& h,
                             const Point& p) const
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
    void extractCoordinates();

    /**
     * @brief
     * @param
     * @return
     */
    bool isNeighbor(const Point* atom1,
                    const Point* atom2);

    MDLabel* d_lb;                     //!<
    SimulationStateP d_sharedState;    //!<
    SimpleMaterial* d_material;        //!<
    IntegratorType d_integrator;       //!<
    double delt;                     //!<

    vector<const VarLabel*> d_particleState;            //!<
    vector<const VarLabel*> d_particleState_preReloc;   //!<

    // fields specific to non-bonded interaction (LJ Potential)
    string d_coordinateFile;  //!< file with coordinates of all atoms in this MD system
    unsigned int d_numAtoms;  //!< Total number of atoms in this MD simulation
    double d_cutoffRadius;    //!< The short ranged cut off distances (in Angstroms)
    Vector d_box;             //!< The size of simulation
    double R12;               //!< This is the v.d.w. repulsive parameter
    double R6;                //!< This is the v.d.w. attractive parameter

    // neighborList[i] contains the index of all atoms located within a short ranged cut off from atom "i"
    std::vector<Point> d_atomList;             //!<
    std::vector<vector<int> > d_neighborList;  //!<

    Electrostatics* d_electrostatics;          //!<
    MDSystem* d_system;                        //!<

    // copy constructor and operator=
    MD(const MD&);                           //!<
    MD& operator=(const MD&);                //!<
};
}

#endif

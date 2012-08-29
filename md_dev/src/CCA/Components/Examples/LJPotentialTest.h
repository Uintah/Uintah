/*

 The MIT License

 Copyright (c) 2012 The University of Utah

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

#ifndef Packages_Uintah_CCA_Components_Examples_LJPotentialTest_h
#define Packages_Uintah_CCA_Components_Examples_LJPotentialTest_h

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <cstdio>
#include <cstring>

namespace Uintah {

  class SimpleMaterial;
  class ExamplesLabel;

  /**************************************

   CLASS
   LJPotentialTest
   
   Lennard Jones Potential Test Simulation

   GENERAL INFORMATION

   LJPotentialTest.h

   Alan Humphrey
   Scientific Computing and Imaging Institute, University of Utah

   KEYWORDS
   LJPotentialTest

   DESCRIPTION
   This class calculates van der Waals forces in a system of "N" atoms using the Lennard-Jones potential
   as an approximation model. Lennard-Jones potential (also referred to as the L-J potential, 6-12 potential,
   or 12-6 potential) is a simple model that approximates the interaction between a pair of neutral atoms
   or molecules.

   WARNING
   None.

   ****************************************/

  class LJPotentialTest : public UintahParallelComponent, public SimulationInterface {

    public:
      LJPotentialTest(const ProcessorGroup* myworld);

      virtual ~LJPotentialTest();

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

    private:

      inline bool containsAtom(const IntVector &l,
                               const IntVector &h,
                               const Point &p)
      {
        return ((p.x() >= l.x() && p.x() < h.x()) && (p.y() >= l.y() && p.y() < h.y()) && (p.z() >= l.z() && p.z() < h.z()));
      }

      void generateNeighborList();

      void extractCoordinates();

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

      void timeAdvance(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw);
      SimulationStateP sharedState_;
      SimpleMaterial* mymat_;
      LJPotentialTest(const LJPotentialTest&);
      LJPotentialTest& operator=(const LJPotentialTest&);
      double delt_;
      int doOutput_;

      std::vector<std::vector<const VarLabel*> > d_particleState;
      std::vector<std::vector<const VarLabel*> > d_particleState_preReloc;

      const VarLabel* pXLabel;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pEnergyLabel;
      const VarLabel* pEnergyLabel_preReloc;
      const VarLabel* pForceLabel;
      const VarLabel* pForceLabel_preReloc;
      const VarLabel* pParticleIDLabel;
      const VarLabel* pParticleIDLabel_preReloc;
      const VarLabel* vdwEnergyLabel;

      // fields specific to non-bonded interaction (LJ Potential)
      string coordinateFile_;
      unsigned int numAtoms_;
      double cutoffDistance_;  // the short ranged cut off distances (in Angstroms)
      Vector box_;  // the size of simulation
      double R12_;  // this is the v.d.w. repulsive parameter
      double R6_;  // this is the v.d.w. attractive parameter

      // neighborList[i] contains the index of all atoms located within a short ranged cut off from atom "i"
      std::vector<Point> atomList;
      std::vector<vector<int> > neighborList;
  };
}

#endif

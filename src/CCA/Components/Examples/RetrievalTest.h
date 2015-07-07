/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 *
 * ----------------------------------------------------------
 * RetrievalTest.h
 *
 *  Created on: Apr 3, 2015
 *      Author: jbhooper
 */

#ifndef RETRIEVALTEST_H_
#define RETRIEVALTEST_H_

#include <CCA/Ports/SimulationInterface.h>

#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/GridVariable.h>

#include <Core/Grid/Variables/ComputeSet.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace Uintah {
  class SimpleMaterial;
  class RetrievalLabel;


  class patchData
  {
  public:
      enum cellType{interior = 0, face = 1, edge = 2, corner =3};

                          patchData(const SCIRun::IntVector centerLow,
                                    const SCIRun::IntVector centerHigh,
                                    const Patch*            current,
                                    const int               neighbors,
                                    const int               expected);
                         ~patchData() {};

            bool          operator<(const patchData& rhs) const;
    inline  int           getExpectedCells() const
      { return d_expectedCells; }
    inline  int           getNeighborCells() const
      { return d_neighboringCells; }
    inline  double        x() const
      { return d_center.x(); }
    inline  double        y() const
      { return d_center.y(); }
    inline  double        z() const
      { return d_center.z(); }
    inline  int           ID() const
      {
        if (d_patchReference) return d_patchReference->getID();
        return -1;
      }
    friend  std::ostream& operator<<(      std::ostream& os,
                                     const Uintah::patchData& patchOut);
  private:
            int                     d_layer;
            cellType                d_patchType;
    const   Patch*                  d_patchReference;
            int                     d_neighboringCells, d_expectedCells;
            SCIRun::Vector          d_center;
  };


  class RetrievalTest : public UintahParallelComponent, public SimulationInterface
  {
    public:
               RetrievalTest(const ProcessorGroup* world);
      virtual ~RetrievalTest();

      virtual void problemSetup(const ProblemSpecP&     problemSpec,
                                const ProblemSpecP&     restartSpec,
                                      GridP&            grid,
                                      SimulationStateP& simState);

      virtual void scheduleInitialize(const LevelP&     level,
                                            SchedulerP& sched);

      virtual void scheduleComputeStableTimestep(const LevelP&      level,
                                                       SchedulerP&  sched);

      virtual void scheduleRestartInitialize(const LevelP&      level,
                                                   SchedulerP&  sched);

      virtual void scheduleTimeAdvance(const LevelP&        level,
                                             SchedulerP&    sched);

    protected:
      void scheduleRetrieveData(      SchedulerP&   sched,
                                const PatchSet*     patches,
                                const MaterialSet*  materials);

    private:
      void initialize(const ProcessorGroup* pg,
                      const PatchSubset*    patches,
                      const MaterialSubset* materials,
                            DataWarehouse*  oldDW,
                            DataWarehouse*  newDW);

      void computeStableTimestep(const ProcessorGroup* pg,
                                 const PatchSubset*    patches,
                                 const MaterialSubset* materials,
                                       DataWarehouse*  oldDW,
                                       DataWarehouse*  newDW);

      void retrieveData(const ProcessorGroup* pg,
                        const PatchSubset*    patches,
                        const MaterialSubset* materials,
                              DataWarehouse*  oldDW,
                              DataWarehouse*  newDW);

      void registerParticleState();

      void outputVectors(const int                          numPerLine,
                               std::ofstream&               outFile,
                               std::vector<SCIRun::Vector>& dataSet);

      void outputIntVectors(const int                             numPerLine,
                                  std::ofstream&                  outFIle,
                                  std::vector<SCIRun::IntVector>& dataSet);

      SimulationStateP  d_sharedState;
      SimpleMaterial    *referenceMaterial;
      double            del_t;

      std::string       d_outfileName;

      std::vector<const VarLabel*> d_particleState;
      std::vector<const VarLabel*> d_particleState_preReloc;

      // Particle variables
      const VarLabel *pX, *pX_preReloc;
      const VarLabel *pLocation, *pLocation_preReloc;
      const VarLabel *pID, *pID_preReloc;

      // Number trackers for ghosted quantities
      int d_numGhostCells;
      int d_CellsPerPatch;

      SCIRun::IntVector d_centerLow, d_centerHigh;

      // copy constructor and assignment operator
      RetrievalTest(const RetrievalTest&);
      RetrievalTest& operator=(const RetrievalTest&);
  };
}


#endif /* RETRIEVALTEST_H_ */

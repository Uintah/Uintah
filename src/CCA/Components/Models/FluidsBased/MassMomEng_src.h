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


#ifndef Packages_Uintah_CCA_Components_Examples_MassMomEng_src_h
#define Packages_Uintah_CCA_Components_Examples_MassMomEng_src_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <Core/GeometryPiece/GeometryPiece.h>

#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class ICELabel;

/**************************************

CLASS
   MassMomEng_src

   MassMomEng_src simulation

GENERAL INFORMATION

   MassMomEng_src.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   MassMomEng_src

DESCRIPTION
   Long description...

WARNING

****************************************/

  class MassMomEng_src : public FluidsBasedModel {
  public:
    MassMomEng_src(const ProcessorGroup* myworld,
                   const MaterialManagerP& materialManager,
                   const ProblemSpecP& params);

    virtual ~MassMomEng_src();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid,
                               const bool isRestart);

    virtual void scheduleInitialize(SchedulerP&,
                                    const LevelP& level);

    virtual void scheduleRestartInitialize(SchedulerP&,
                                           const LevelP& level){};

    virtual void scheduleComputeStableTimeStep(SchedulerP&,
                                               const LevelP& level);

    virtual void scheduleComputeModelSources(SchedulerP&,
                                             const LevelP& level);

    virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                                         const LevelP&,
                                                         const MaterialSet*);

    virtual void computeSpecificHeat(CCVariable<double>&,
                                     const Patch*,
                                     DataWarehouse*,
                                     const int);

   virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched);

   virtual void scheduleTestConservation(SchedulerP&,
                                         const PatchSet* patches);

  private:
    void computeModelSources(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse*,
                             DataWarehouse* new_dw);

    MassMomEng_src(const MassMomEng_src&);
    MassMomEng_src& operator=(const MassMomEng_src&);

    ProblemSpecP d_params;
    ICELabel* Ilb;
    MaterialSet* d_matlSet;
    Material* d_matl;

    //__________________________________
    //  Region used for initialization
    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);

      enum algorithm {fixedPrimitiveValues};
      algorithm algoType;

      GeometryPieceP piece;
      Vector velocity_src;
      double density_src;
      double temp_src;
      double timeStart;
      double timeStop;
    };

    class src{
    public:
      std::vector<MassMomEng_src::Region*> regions;
    };

    src* d_src;

    const VarLabel* totalMass_srcLabel;
    const VarLabel* totalMom_srcLabel;
    const VarLabel* totalEng_srcLabel;
  };
}

#endif

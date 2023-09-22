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


#ifndef Packages_Uintah_CCA_Components_Examples_BinaryProperties_h
#define Packages_Uintah_CCA_Components_Examples_BinaryProperties_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <map>
#include <vector>

namespace Uintah {
  class ICELabel;

/**************************************

CLASS
   BinaryProperties

   BinaryProperties simulation

GENERAL INFORMATION

   BinaryProperties.h

   Todd Harman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   BinaryProperties

DESCRIPTION
   This model will vary the material properties in a single matl ICE
   simulation using a simple binary mixture fraction approach.  This
   is mainly for debugging.
WARNING

****************************************/

  class BinaryProperties :public FluidsBasedModel {
  public:
    BinaryProperties(const ProcessorGroup* myworld,
              const MaterialManagerP& materialManager,
              const ProblemSpecP& params);

    virtual ~BinaryProperties();

    virtual void outputProblemSpec(ProblemSpecP& ps){};

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
    ICELabel* Ilb;

    void modifyThermoTransportProperties(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset*,
                                         DataWarehouse*,
                                         DataWarehouse* new_dw);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse*,
                    DataWarehouse* new_dw);

    void computeModelSources(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw);

    void testConservation(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);
    //__________________________________
    BinaryProperties(const BinaryProperties&);
    BinaryProperties& operator=(const BinaryProperties&);

    ProblemSpecP d_params {nullptr};

    const Material* d_matl;
    MaterialSet* d_matl_set;

    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);

      GeometryPieceP piece;
      double initialScalar;
    };

    class Scalar {
    public:
      int index;
      std::string name;
      // labels for this particular scalar
      VarLabel* scalar_CCLabel;
      VarLabel* source_CCLabel;
      VarLabel* diffusionCoefLabel;

      std::vector<Region*> regions;
      double diff_coeff;
      int  initialize_diffusion_knob;
    };

    // general labels
    class BinaryPropertiesLabel {
    public:
      VarLabel* sum_scalar_fLabel;
    };

    BinaryPropertiesLabel* Slb;
    Scalar* d_scalar;
    double d_rho_A;
    double d_rho_B;
    double d_cv_A;
    double d_cv_B;
    double d_R_A;
    double d_R_B;
    double d_thermalCond_A;
    double d_thermalCond_B;
    double d_viscosity_A;
    double d_viscosity_B;
    bool d_test_conservation;
  };
}

#endif

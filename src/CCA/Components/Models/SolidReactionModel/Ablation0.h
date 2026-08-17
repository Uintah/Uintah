/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_Models_Ablation0_h
#define Packages_Uintah_CCA_Components_Models_Ablation0_h

#include <CCA/Ports/ModelInterface.h>

namespace Uintah {

  class RateConstant;
  class RateModel;

  class ICELabel;
  class MPMLabel;

  /**************************************

     CLASS
     Ablation0

     A generalized Rate Model class that very much borrows
     from the factory model idiom for rate expression composition.

     GENERAL INFORMATION

     Ablation0.h

     Jim Guilkey
     Department of Mechanical Engineering
     University of Utah

     KEYWORDS
     Ablation

     DESCRIPTION
     Constant Rate Surface Ablation to get the steps working...

     WARNING

     ****************************************/

    class Ablation0 : public ModelInterface {
    public:
      Ablation0(const ProcessorGroup* d_myworld,
                         const MaterialManagerP& materialManager,
                         const ProblemSpecP& params,
                         const ProblemSpecP& prob_spec);

      virtual ~Ablation0();

      virtual void problemSetup(GridP& grid, const bool isRestart);

      virtual void outputProblemSpec(ProblemSpecP& ps);

      virtual void scheduleInitialize(SchedulerP&,
                                      const LevelP& level);

      virtual void scheduleRestartInitialize(SchedulerP&,
                                             const LevelP& level){};

      virtual void scheduleComputeStableTimeStep(SchedulerP& sched,
                                                 const LevelP& level);

      virtual void scheduleComputeModelSources(SchedulerP&,
                                                 const LevelP& level);

    private:

      void computeModelSources(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw);

      // Functions
      Ablation0(const Ablation0&);
      Ablation0& operator=(const Ablation0&);

      // Innards
      RateConstant * d_rateConstantModel {nullptr};     // k(T)
      RateModel    * d_rateModel         {nullptr};     // f(a)

      const Material* d_reactant  {nullptr};
      const Material* d_product   {nullptr};

      std::string d_fromMaterial;
      std::string d_doMaterial;
      double d_E0;                            // Enthalpy change for reaction in J/kg

      ICELabel *Ilb;                          // Used to get handles on temperature, pressure, etc.
      MPMLabel *Mlb;                          // Used to get handles on particle data.
      MaterialSet *d_myMatls;                   // All the materials referenced by this model

      // Variables used for tracking the Reaction
      const VarLabel* reactedFractionLabel;   // Fraction of reactant in cell
      const VarLabel* delFLabel;              // Change of fraction of reactant during timestep
      const VarLabel* totalMassBurnedLabel;
      const VarLabel* totalHeatReleasedLabel;

      // flags for the conservation test
      struct saveConservedVars{
        bool onOff;
        bool mass;
        bool energy;
      };

      saveConservedVars* d_saveConservedVars;

      // Some Uintah Necessities
      ProblemSpecP d_params {nullptr};
      ProblemSpecP d_prob_spec {nullptr};
    };
}

#endif

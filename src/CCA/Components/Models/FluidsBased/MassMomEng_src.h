/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Packages_Uintah_CCA_Components_Examples_MassMomEng_src_h
#define Packages_Uintah_CCA_Components_Examples_MassMomEng_src_h

#include <CCA/Ports/ModelInterface.h>
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

  class MassMomEng_src : public ModelInterface {
  public:
    MassMomEng_src(const ProcessorGroup* myworld, ProblemSpecP& params);
    virtual ~MassMomEng_src();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
                              ModelSetup* setup);
      
    virtual void scheduleInitialize(SchedulerP&,
                                        const LevelP& level,
                                        const ModelInfo*);

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimestep(SchedulerP&,
                                                   const LevelP& level,
                                                   const ModelInfo*);
      
    virtual void scheduleComputeModelSources(SchedulerP&,
                                                const LevelP& level,
                                                const ModelInfo*);
                                             
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
                                         const PatchSet* patches,
                                         const ModelInfo* mi);

  private:    
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset* patches,
                               const MaterialSubset* matls, 
                             DataWarehouse*, 
                               DataWarehouse* new_dw, 
                             const ModelInfo*);

    MassMomEng_src(const MassMomEng_src&);
    MassMomEng_src& operator=(const MassMomEng_src&);

    ProblemSpecP params;
    ICELabel* Ilb;
    MaterialSet* mymatls;
    Material* d_matl;
    
    struct src{
      Vector mom_src_rate;
      double mass_src_rate;
      double eng_src_rate;
      double d_mme_src_t_start;
      double d_mme_src_t_final;
    };
    src* d_src;
    
    const VarLabel* totalMass_srcLabel;
    const VarLabel* totalMom_srcLabel;
    const VarLabel* totalEng_srcLabel;
    SimulationStateP d_sharedState;
  };
}

#endif

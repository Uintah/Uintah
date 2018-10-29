/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_HOMEBREW_FluidsBasedModel_H
#define UINTAH_HOMEBREW_FluidsBasedModel_H

#include <CCA/Ports/ModelInterface.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>


/**************************************

CLASS
   FluidsBasedModel
   
   Short description...

GENERAL INFORMATION

   FluidsBasedModel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Model of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Model_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

namespace Uintah {

  class ApplicationInterface;
  class Regridder;
  class Output;

  class DataWarehouse;
  class Material;
  class ProcessorGroup;
  class VarLabel;


  struct TransportedVariable {
    const MaterialSubset* matls;
    const MaterialSet* matlSet;
    const VarLabel* var;
    const VarLabel* src;
    const VarLabel* var_Lagrangian;
    const VarLabel* var_adv;
  };
      
  struct AMRRefluxVariable {
    const MaterialSubset* matls;
    const MaterialSet* matlSet;
    const VarLabel* var;
    const VarLabel* var_adv;
    const VarLabel* var_X_FC_flux;
    const VarLabel* var_Y_FC_flux;
    const VarLabel* var_Z_FC_flux;

    const VarLabel* var_X_FC_corr;
    const VarLabel* var_Y_FC_corr;
    const VarLabel* var_Z_FC_corr;
  };
  
  //________________________________________________
  class FluidsBasedModel : public ModelInterface {
  public:
    FluidsBasedModel(const ProcessorGroup* myworld,
                     const MaterialManagerP materialManager);

    virtual ~FluidsBasedModel();

    virtual void problemSetup(GridP& grid,
                               const bool isRestart) = 0;
      
    virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

    virtual void scheduleInitialize(SchedulerP& scheduler,
                                    const LevelP& level) = 0;

      
    virtual void scheduleComputeStableTimeStep(SchedulerP& scheduler,
                                               const LevelP& level) = 0;
      
    virtual void scheduleComputeModelSources(SchedulerP& scheduler,
                                             const LevelP& level) = 0;
                                                
    virtual void computeSpecificHeat(CCVariable<double>&,
                                     const Patch* patch,
                                     DataWarehouse* new_dw,
                                     const int indx) = 0;
                                     
    virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched) = 0;
                                               
    virtual void scheduleTestConservation(SchedulerP&,
                                          const PatchSet* patches) = 0;

    virtual void scheduleModifyThermoTransportProperties(SchedulerP& scheduler,
                                                         const LevelP& level,
                                                         const MaterialSet*) = 0;

    // Method specific to FluidsBasedModels
    virtual void registerTransportedVariable(const MaterialSet* matlSet,
                                             const VarLabel* var,
                                             const VarLabel* src);
                                        
    virtual void registerAMRRefluxVariable(const MaterialSet* matlSet,
                                           const VarLabel* var);

    virtual std::vector<TransportedVariable*> getTransportedVars() {return d_trans_vars; }
    virtual std::vector<AMRRefluxVariable*> getAMRRefluxVars() { return d_reflux_vars; }

    virtual bool computesThermoTransportProps() const
    { return m_modelComputesThermoTransportProps; }

  // protected:
    std::vector<TransportedVariable*> d_trans_vars;
    std::vector<AMRRefluxVariable*> d_reflux_vars;

    bool m_modelComputesThermoTransportProps {false};

  private:
    FluidsBasedModel(const FluidsBasedModel&);
    FluidsBasedModel& operator=(const FluidsBasedModel&);
  };
} // End namespace Uintah
   
#endif

/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_ModelInterface_H
#define UINTAH_HOMEBREW_ModelInterface_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Util/Handle.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/Variables/CCVariable.h>


namespace Uintah {
/**************************************

CLASS
   ModelInterface
   
   Short description...

GENERAL INFORMATION

   ModelInterface.h

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

  class DataWarehouse;
  class Material;
  class ProcessorGroup;
  class VarLabel;
  class ModelSetup {
    public:
    virtual void registerTransportedVariable(const MaterialSet* matlSet,
					     const VarLabel* var,
					     const VarLabel* src) = 0;
                                        
    virtual void registerAMR_RefluxVariable(const MaterialSet* matlSet,
					         const VarLabel* var) = 0;

    virtual ~ModelSetup() {};
  };
  class ModelInfo {
  public:
    ModelInfo(const VarLabel* delt, 
	      const VarLabel* mass_source,
	      const VarLabel* momentum_source, 
	      const VarLabel* energy_source,
	      const VarLabel* sp_vol_source,
	      const VarLabel* density, 
	      const VarLabel* velocity,
	      const VarLabel* temperature, 
	      const VarLabel* pressure,
	      const VarLabel* specificVol,
             const VarLabel* specific_heat,
             const VarLabel* gamma)
      : delT_Label(delt), 
        modelMass_srcLabel(mass_source),
        modelMom_srcLabel(momentum_source),
        modelEng_srcLabel(energy_source),
        modelVol_srcLabel(sp_vol_source),
        rho_CCLabel(density), 
        vel_CCLabel(velocity),
        temp_CCLabel(temperature), 
        press_CCLabel(pressure),
        sp_vol_CCLabel(specificVol),
        specific_heatLabel(specific_heat),
        gammaLabel(gamma)
      {
      }
    const VarLabel* delT_Label;

    const VarLabel* modelMass_srcLabel;
    const VarLabel* modelMom_srcLabel;
    const VarLabel* modelEng_srcLabel;
    const VarLabel* modelVol_srcLabel;
    const VarLabel* rho_CCLabel;
    const VarLabel* vel_CCLabel;
    const VarLabel* temp_CCLabel;
    const VarLabel* press_CCLabel;
    const VarLabel* sp_vol_CCLabel;
    const VarLabel* specific_heatLabel;
    const VarLabel* gammaLabel;
  private:
    ModelInfo(const ModelInfo&);
    ModelInfo& operator=(const ModelInfo&);
  };  // class ModelInfo
  
  
   //________________________________________________
   class ModelInterface : public UintahParallelPort {
   public:
     ModelInterface(const ProcessorGroup* d_myworld);
     virtual ~ModelInterface();

     virtual void outputProblemSpec(ProblemSpecP& ps) = 0;
      
     virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			       ModelSetup* setup) = 0;
      
     virtual void activateModel(GridP& grid, SimulationStateP& sharedState,
			        ModelSetup* setup);
      
     virtual void scheduleInitialize(SchedulerP&,
				     const LevelP& level,
				     const ModelInfo*) = 0;

     virtual void restartInitialize() {}
      
     virtual void scheduleComputeStableTimestep(SchedulerP& sched,
						const LevelP& level,
						const ModelInfo*) = 0;
      
     virtual void scheduleComputeModelSources(SchedulerP&,
						    const LevelP& level,
						    const ModelInfo*) = 0;
                                              
     virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                                const LevelP&,
                                                const MaterialSet*) = 0;
                                                
     virtual void computeSpecificHeat(CCVariable<double>&,
                                     const Patch* patch,
                                     DataWarehouse* new_dw,
                                     const int indx) = 0;
                                     
     virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                       SchedulerP& sched) =0;                                  
    
     virtual void scheduleCheckNeedAddMaterial(SchedulerP&,
                                               const LevelP& level,
                                               const ModelInfo*);
                                               
     virtual void scheduleTestConservation(SchedulerP&,
                                           const PatchSet* patches,
                                           const ModelInfo* mi)=0;    

     virtual void setMPMLabel(MPMLabel* MLB);
                                               
    bool computesThermoTransportProps() const;
    bool d_modelComputesThermoTransportProps;
    Output* d_dataArchiver;
   
   protected:
     const ProcessorGroup* d_myworld;
   private:
     
     ModelInterface(const ModelInterface&);
     ModelInterface& operator=(const ModelInterface&);
   };
} // End namespace Uintah
   


#endif

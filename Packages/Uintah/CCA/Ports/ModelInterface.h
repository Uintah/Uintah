#ifndef UINTAH_HOMEBREW_ModelInterface_H
#define UINTAH_HOMEBREW_ModelInterface_H

#include <Packages/Uintah/Core/Parallel/UintahParallelPort.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Util/Handle.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/Output.h>

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>

#include <Packages/Uintah/CCA/Ports/uintahshare.h>

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
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Model_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  using namespace std;
  class DataWarehouse;
  class Material;
  class ProcessorGroup;
  class VarLabel;
  class ModelSetup {
    public:
    virtual void registerTransportedVariable(const MaterialSubset* matls,
					     const VarLabel* var,
					     const VarLabel* src) = 0;
                                        
    virtual void registerAMR_RefluxVariable(const MaterialSubset* matls,
					         const VarLabel* var) = 0;

    virtual ~ModelSetup() {};
  };
  class UINTAHSHARE ModelInfo {
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
   class UINTAHSHARE ModelInterface : public UintahParallelPort {
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

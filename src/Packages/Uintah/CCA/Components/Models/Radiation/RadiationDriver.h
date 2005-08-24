//----- RadiationDriver.h ----------------------------------------------

#ifndef Uintah_Component_Models_RadiationDriver_h
#define Uintah_Component_Models_RadiationDriver_h

/**************************************
CLASS
   RadiationDriver
   
   Class RadiationDriver is the driver for the different radiation
   models

GENERAL INFORMATION
   RadiationDriver.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date: April 12, 2005
   
   C-SAFE 
   
   Copyright U of U 2005

KEYWORDS


DESCRIPTION
   This driver performs the tasks that EnthalpySolver used to perform
   in Arches insofar as the radiation tasks are concerned.  This makes it
   modular and independent of ICE (directly; it still may use temperature
   from ICE)

WARNING
   none

************************************************************************/

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationVariables.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationConstVariables.h>

namespace Uintah {
  class ProcessorGroup;
  class Models_RadiationModel;

class RadiationDriver : public ModelInterface {

 public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of the Radiation driver.
  // PRECONDITIONS
  // POSTCONDITIONS

  RadiationDriver(const ProcessorGroup* myworld, ProblemSpecP& params);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
  virtual ~RadiationDriver();

  ////////////////////////////////////////////////////////////////////////
  // GROUP: Schedule Action
  ////////////////////////////////////////////////////////////////////////

  virtual void problemSetup(GridP& grid,
                            SimulationStateP& sharedState,
                            ModelSetup* setup);

  virtual void scheduleInitialize(SchedulerP& sched,
                                  const LevelP& level,
                                  const ModelInfo*);

  virtual void restartInitialize() {}
      
  virtual void scheduleComputeStableTimestep(SchedulerP&,
                                             const LevelP& level,
                                             const ModelInfo*);
                                  
  virtual void scheduleComputeModelSources(SchedulerP& sched, 
                                           const LevelP& level,
                                           const ModelInfo*);

  void scheduleComputeCO2_H2O(const LevelP& level,
                              SchedulerP& sched,
                              const PatchSet* patches,
                              const MaterialSet* matls);
                                                                                                       
  void scheduleCopyValues(const LevelP& level,
                          SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSet* matls);

  void scheduleComputeProps(const LevelP& level,
                            SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* matls);

  void scheduleBoundaryCondition(const LevelP& level,
                                 SchedulerP& sched,
                                 const PatchSet* patches,
                                 const MaterialSet* matls);

  virtual void scheduleIntensitySolve(const LevelP& level,
                              SchedulerP& sched,
                              const PatchSet* patches,
                              const MaterialSet* matls,
                              const ModelInfo* mi);

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

  VarLabel* d_cellInfoLabel;

  VarLabel* shgamma_CCLabel;
  VarLabel* abskg_CCLabel;
  VarLabel* esrcg_CCLabel;  
  VarLabel* cenint_CCLabel;  

  VarLabel* cellType_CCLabel;

  VarLabel* qfluxE_CCLabel;
  VarLabel* qfluxW_CCLabel;
  VarLabel* qfluxN_CCLabel;
  VarLabel* qfluxS_CCLabel;
  VarLabel* qfluxT_CCLabel;
  VarLabel* qfluxB_CCLabel;

  VarLabel* co2_CCLabel;
  VarLabel* h2o_CCLabel;
  VarLabel* radCO2_CCLabel;
  VarLabel* radH2O_CCLabel;
  VarLabel* mixfrac_CCLabel;
  VarLabel* mixfracCopy_CCLabel;
  VarLabel* density_CCLabel;
  VarLabel* iceDensity_CCLabel;
  VarLabel* temp_CCLabel;
  VarLabel* iceTemp_CCLabel;
  VarLabel* tempCopy_CCLabel;
  VarLabel* sootVF_CCLabel;
  VarLabel* sootVFCopy_CCLabel;

  VarLabel* scalar_CCLabel;
  
  // Need to change the label below to a public source term for
  // use in ICE
  VarLabel* radiationSrc_CCLabel;

 private:

  Models_RadiationModel* d_DORadiation;
  int d_radCounter; //to decide how often radiation calc is done
  int d_radCalcFreq;
  bool d_useIceTemp;
  bool d_useTableValues;
  bool d_computeCO2_H2O_from_f;
  
  const PatchSet* d_perproc_patches;

  ProblemSpecP params;

  const ProcessorGroup* d_myworld;
  SimulationStateP d_sharedState;

  void initialize(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse*,
                  DataWarehouse* new_dw);

  void buildLinearMatrix(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse*,
                         DataWarehouse*);
                         
  void computeCO2_H2O(const ProcessorGroup*, 
                      const PatchSubset* patches,
                      const MaterialSubset*,
                      DataWarehouse*,
                      DataWarehouse* new_dw);
                      
  void copyValues(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);

  void computeProps(const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* ,
                    DataWarehouse* new_dw);

  void boundaryCondition(const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* ,
                         DataWarehouse* new_dw);

  void intensitySolve(const ProcessorGroup* pc,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* ,
                      DataWarehouse* new_dw,
                      const ModelInfo* mi);

  void modifyThermoTransportProperties(const ProcessorGroup*, 
                                       const PatchSubset* patches,        
                                       const MaterialSubset*,             
                                       DataWarehouse*,                    
                                       DataWarehouse* new_dw);             

}; // End class RadiationDriver
} // End namespace Uintah


#endif


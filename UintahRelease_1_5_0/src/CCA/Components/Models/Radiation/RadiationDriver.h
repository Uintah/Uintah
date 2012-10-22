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
   
   
KEYWORDS


DESCRIPTION
   This driver performs the tasks that EnthalpySolver used to perform
   in Arches insofar as the radiation tasks are concerned.  This makes it
   modular and independent of ICE (directly; it still may use temperature
   from ICE)

WARNING
   none

************************************************************************/

#include <CCA/Ports/SchedulerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/ModelInterface.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Components/Models/Radiation/RadiationVariables.h>
#include <CCA/Components/Models/Radiation/RadiationConstVariables.h>

namespace Uintah {

class ProcessorGroup;
class Models_RadiationModel;
class ICELabel;
  
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

  virtual void outputProblemSpec(ProblemSpecP& ps);

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
                              
  void scheduleSet_cellType(const LevelP& level,
                            SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSubset* mss_G,
                            const MaterialSubset* mss_S,
                            const MaterialSet* matls_set_GS);

  void scheduleCopyValues(const LevelP& level,
                          SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSubset* mss_G,
                          const MaterialSubset* mss_S,
                          const MaterialSet* matls_set_GS);

  void scheduleComputeProps(const LevelP& level,
                            SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSubset* mss_G,
                            const MaterialSubset* mss_S,
                            const MaterialSet* matls_set_GS);

  void scheduleBoundaryCondition(const LevelP& level,
                                 SchedulerP& sched,
                                 const PatchSet* patches,
                                 const MaterialSet* matls);

  virtual void scheduleIntensitySolve(const LevelP& level,
                                      SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSubset* mss_G,
                                      const MaterialSubset* mss_S,
                                      const MaterialSet* matls_set_GS,
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
  VarLabel* radiationSrc_CCLabel;
  VarLabel* solidEmissionLabel;
  VarLabel* isGasSolidInterfaceLabel;
  VarLabel* insideSolidLabel;

 private:

  Models_RadiationModel* d_DORadiation;
  int d_radCounter; //to decide how often radiation calc is done
  int d_radCalcFreq;
  double d_radCalc_interval;
  double d_radCalc_nextTime;
  double d_sigma;
  bool d_doRadCalc;
  bool d_useTableValues;
  bool d_hasAbsorbingSolid;
  string d_table_or_ice_temp_density;
  
  const PatchSet* d_perproc_patches;

  ICELabel* Ilb;

  ProblemSpecP params;
  const Material* d_matl_G; //gas
  const Material* d_matl_S; //solid

  const ProcessorGroup* d_myworld;
  SimulationStateP d_sharedState;

  vector<GeometryPieceP> d_geom_pieces;

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
                      
  void set_cellType(const ProcessorGroup*, 
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

  void solidEmission(CCVariable<double>& energySrc_solid,
                     constCCVariable<double>& vol_frac_solid,
                     CCVariable<double>& solidEmission,
                     const double delT,
                     const Patch* patch,                        
                     DataWarehouse* new_dw);
                
  void modifyThermoTransportProperties(const ProcessorGroup*, 
                                       const PatchSubset* patches,        
                                       const MaterialSubset*,             
                                       DataWarehouse*,                    
                                       DataWarehouse* new_dw);         

}; // End class RadiationDriver
} // End namespace Uintah


#endif


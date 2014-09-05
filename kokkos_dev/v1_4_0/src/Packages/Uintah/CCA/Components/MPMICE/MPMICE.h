#ifndef UINTAH_HOMEBREW_MPMICE_H
#define UINTAH_HOMEBREW_MPMICE_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>

#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICELabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>

#include <Core/Geometry/Vector.h>

namespace Uintah {
  class Output;

using namespace SCIRun;

/**************************************

CLASS
   MPMICE
   
   Short description...

GENERAL INFORMATION

   MPMICE.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MPMICE

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class MPMICE : public UintahParallelComponent, public SimulationInterface {
public:
  MPMICE(const ProcessorGroup* myworld);
  virtual ~MPMICE();
         
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
                            SimulationStateP&);
         
  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);

  virtual void restartInitialize();

  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP&);
         
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);

                            
  void scheduleInterpolateNCToCC_0(SchedulerP&, 
                                  const PatchSet*,
                                  const MaterialSubset*,
                                  const MaterialSet*);

  
  void scheduleComputeLagrangianValuesMPM(SchedulerP&, 
                                          const PatchSet*,
                                          const MaterialSubset*,
                                          const MaterialSet*);

  void scheduleCCMomExchange(SchedulerP&, 
                            const PatchSet*,
                            const MaterialSubset*,
                            const MaterialSubset*,
                            const MaterialSet*);

  void scheduleInterpolateCCToNC(SchedulerP&, const PatchSet*,
                                 const MaterialSet*);

  void scheduleComputeNonEquilibrationPressure(SchedulerP&, 
                                               const PatchSet*,
                                               const MaterialSubset*,
                                               const MaterialSubset*,
                                               const MaterialSubset*,
                                               const MaterialSet*);

  void scheduleComputePressure(SchedulerP&, 
                               const PatchSet*,
                               const MaterialSubset*,
                               const MaterialSubset*,
                               const MaterialSubset*,
                               const MaterialSet*);


  void scheduleInterpolatePressCCToPressNC(SchedulerP&, 
                                          const PatchSet*,
                                          const MaterialSubset*,
                                          const MaterialSet*);

  void scheduleInterpolatePAndGradP(SchedulerP&, 
                                    const PatchSet*,
                                    const MaterialSubset*,
                                    const MaterialSubset*,
                                    const MaterialSubset*,
                                    const MaterialSet*);

  void scheduleHEChemistry(SchedulerP&, 
                           const PatchSet*,
                           const MaterialSubset*,
                           const MaterialSubset*,
                           const MaterialSubset*,
                           const MaterialSet*);
  
  void scheduleInterpolateMassBurnFractionToNC( SchedulerP&, 
                                                const PatchSet*,
                                                const MaterialSet*);            
//______________________________________________________________________
//       A C T U A L   S T E P S : 
  void actuallyInitialize(const ProcessorGroup*,
                          const PatchSubset* patch,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse* new_dw);
                         
                                                    
  void interpolateNCToCC_0(const ProcessorGroup*,
                           const PatchSubset* patch,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);
  
  void computeLagrangianValuesMPM(const ProcessorGroup*,
                                  const PatchSubset* patch,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  void computeEquilibrationPressure(const ProcessorGroup*,
                                    const PatchSubset* patch,
                                    const MaterialSubset* matls,
                                    DataWarehouse*, 
                                    DataWarehouse*);

  void doCCMomExchange(const ProcessorGroup*,
                       const PatchSubset* patch,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw);

  void interpolateCCToNC(const ProcessorGroup*,
                         const PatchSubset* patch,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw);

  void interpolatePressCCToPressNC(const ProcessorGroup*,
                                   const PatchSubset* patch,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  void interpolatePAndGradP(const ProcessorGroup*,
                            const PatchSubset* patch,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw);
                            
  void HEChemistry(const ProcessorGroup*,
                     const PatchSubset* patch,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);
  
  void interpolateMassBurnFractionToNC(const ProcessorGroup*,
                                       const PatchSubset* patch,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);
  void printData(const Patch* patch, 
                  int   include_EC,
                  const string&    message1,        
                  const string&    message2,  
                  const NCVariable<double>& q_NC);
                  
  void printNCVector(const Patch* patch, int include_EC,
                     const string&    message1,        
                     const string&    message2, 
                     int     component,
                     const NCVariable<Vector>& q_NC);
                     
//__________________________________
//    R A T E   F O R M                   
  void computeRateFormPressure(const ProcessorGroup*,
                               const PatchSubset* patch,
                               const MaterialSubset* matls,
                               DataWarehouse*, 
                               DataWarehouse*); 
     
  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

protected:
  MPMICE(const MPMICE&);
  MPMICE& operator=(const MPMICE&);
  SimulationStateP d_sharedState; 
  Output* dataArchiver;
  MPMLabel* Mlb;
  ICELabel* Ilb;
  MPMICELabel* MIlb;
  bool             d_burns;
  SerialMPM*       d_mpm;
  ICE*             d_ice;
  bool             d_fracture;

  double d_dbgTime; 
  double d_dbgStartTime;
  double d_dbgStopTime;
  double d_dbgOutputInterval;
  double d_dbgNextDumpTime;
  double d_dbgOldTime;
  IntVector d_dbgBeginIndx;
  IntVector d_dbgEndIndx; 
  int d_dbgSigFigs;

  vector<MPMPhysicalBC*> d_physicalBCs;
  double d_SMALL_NUM;
  
  // Debugging switches
  bool switchDebug_InterpolateNCToCC_0;
  bool switchDebug_InterpolateCCToNC;
};

} // End namespace Uintah
      
#endif

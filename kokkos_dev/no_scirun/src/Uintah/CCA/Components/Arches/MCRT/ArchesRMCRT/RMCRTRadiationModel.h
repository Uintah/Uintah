// RMCRTRadiationModel.h-------------------------------------------------------
// Reverse Monte Carlo Ray Tracing Radiation Model interface
// 
// (other comments here) 
// 

#ifndef Uintah_Component_Arches_RMCRTRadiationModel_h
#define Uintah_Component_Arches_RMCRTRadiationModel_h

#include <Uintah/Core/Grid/LevelP.h>
#include <Uintah/CCA/Ports/SimulationInterface.h>
#include <Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {

class RMCRTRadiationModel {

public:
  // constructor
  RMCRTRadiationModel( const ArchesLabel* label );

  // destructor
  ~RMCRTRadiationModel();
  
  //methods: 
  /** @brief Set any parameters from the input file, initialize constants, etc... */
  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Schedule the solution of the radiative transport equation using RMCRT */
  void sched_solve( const LevelP& level, SchedulerP& sched );

  /** @brief Actually solve the radiative transport equation using RMCRT */
  void solve( const ProcessorGroup* pc,  
              const PatchSubset* patches, 
              const MaterialSubset*, 
              DataWarehouse* old_dw, 
              DataWarehouse* new_dw );
 

private:


  // variables:
  int d_constNumRays;
  const ArchesLabel* d_lab; 

}; // end class RMCRTRadiationModel
} // end uintah namespace

#endif

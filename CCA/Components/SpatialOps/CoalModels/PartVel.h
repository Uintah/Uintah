#ifndef PartVel_h
#define PartVel_h

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Ports/Scheduler.h>

#include <map>
#include <string>
#include <iostream>

#define YDIM
//#define ZDIM

//===========================================================================

namespace Uintah {
class Fields; 
class  PartVel {
   
public:

  PartVel(Fields* fieldLabels);
 
  ~PartVel();
  /** @brief Interface to the input file */
  void problemSetup( const ProblemSpecP& inputdb ); 
  /** @brief Schedules the calculation of the particle velocities */
  void schedComputePartVel( const LevelP& level, SchedulerP& sched, const int rkStep );
  /** @brief Actually computes the particle velocities */ 
  void ComputePartVel( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw, const int rkStep );

private:

  Fields* d_fieldLabels; 
  
  // velocity model paramters
  double Re; 
  double eta; 
  double rhof; 
  double beta; 
  double eps; 
  double partMass; 
  int regime; 

 }; //end class Fields

} //end namespace Uintah
#endif 

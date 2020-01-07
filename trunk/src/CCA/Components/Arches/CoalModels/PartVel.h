#ifndef PartVel_h
#define PartVel_h
#include <CCA/Ports/Scheduler.h>
#include <iostream>

//===========================================================================

/** @class    PartVel
  * @author   Jeremy Thornock
  *
  * @brief    This model calculates a particle velocity, given information about
  *           the particle, using the fast equilibirum Eulerian approximation
  *           detailed in Balachandar (2008) and Ferry and Balachandar (2001, 2003).
  */

namespace Uintah {
class ArchesLabel;
class  PartVel {

public:

  PartVel(ArchesLabel* fieldLabels);

  ~PartVel();

  /** @brief Interface to the input file */
  void problemSetup( const ProblemSpecP& inputdb );

  /** @brief Schedules the initialization of the particle velocities */
  void schedInitPartVel( const LevelP& level, SchedulerP& sched );

  /** @brief Schedules the calculation of the particle velocities */
  void schedComputePartVel( const LevelP& level, SchedulerP& sched, const int rkStep );

  /** @brief Actually computes the particle velocities */
  void InitPartVel( const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw );

  /** @brief Actually computes the particle velocities (removed, skeleton left in case model is reformulated) */
  void ComputePartVel( const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, const int rkStep );


private:

  std::string get_env_name( const std::string base, const int i ){
    std::string name;
    name = base + "_";
    std::stringstream env;
    env << i;
    name += env.str();
    return name;
  }

  ArchesLabel* d_fieldLabels;

  std::map<int, const VarLabel*> _face_x_part_vel_labels;
  std::map<int, const VarLabel*> _face_y_part_vel_labels;
  std::map<int, const VarLabel*> _face_z_part_vel_labels;

  std::string _uname, _vname, _wname;

 }; //end class PartVel

} //end namespace Uintah
#endif

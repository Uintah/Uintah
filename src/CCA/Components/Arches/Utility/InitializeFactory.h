#ifndef UT_InitializeFactory_h
#define UT_InitializeFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class InitializeFactory : public TaskFactoryBase {

  public:

    InitializeFactory( const ApplicationCommon* arches );
    ~InitializeFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset(const std::string subset){

      if ( subset == _all_tasks_str ){
        return _active_tasks;
      } else if ( subset == "phi_tasks" ) {
        // These tasks do not have a density pre-multiplier
        return _unweighted_var_tasks;
      } else if ( subset == "rho_phi_tasks" ) {
        // These tasks DO have a density pre-multiplier
        return _weighted_var_tasks;
      } else {

      throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset, which means there is no implementation for this factory.",__FILE__,__LINE__);
      }
    }

    void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart );


  protected:


  private:

    std::vector<std::string> _unweighted_var_tasks; // these taks do NOT depend on density
    std::vector<std::string> _weighted_var_tasks; // these taks do depend on density



  };
}
#endif

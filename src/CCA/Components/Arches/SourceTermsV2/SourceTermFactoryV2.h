#ifndef UT_SourceTermV2Factory_h
#define UT_SourceTermV2Factory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class SourceTermV2Factory : public TaskFactoryBase {

  public:

    SourceTermV2Factory( );
    ~SourceTermV2Factory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset( const std::string subset ) {

      if ( subset == "pre_update_source_k_task" ){
        return _pre_update_source_k_task;
      }
       else if ( subset == _all_tasks_str ){
        return _active_tasks;

      } else {

        throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset in SourceTermV2Factory, which means there is no specific implementation for this factory.",__FILE__,__LINE__);

      }
    }

    void add_task( ProblemSpecP& db );

    void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart );

  protected:

  private:
    std::vector<std::string> _pre_update_source_k_task;  ///<Tasks that execute at the start of an RK step
//    std::vector<std::string> _task_order;                 ///<The order in which these tasks should execute
  };
}
#endif

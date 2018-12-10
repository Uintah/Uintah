#ifndef UT_SourceTermFactoryV2_h
#define UT_SourceTermFactoryV2_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class SourceTermFactoryV2 : public TaskFactoryBase {

  public:

    SourceTermFactoryV2( const ApplicationCommon* arches );
    ~SourceTermFactoryV2();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset( const std::string subset ) {

      if ( subset == "pre_update_source_tasks" ){

        return _pre_update_source_tasks;

      } else if ( subset == _all_tasks_str ){

        return _active_tasks;

      } else {

        throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset in SourceTermFactoryV2, which means there is no specific implementation for this factory.",__FILE__,__LINE__);

      }
    }

    void add_task( ProblemSpecP& db );

    void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart );

  private:

    std::vector<std::string> _pre_update_source_tasks;  ///<Tasks that execute at the start of an RK step

  };
}
#endif

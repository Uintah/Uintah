#ifndef UT_ChemMixFactory_h
#define UT_ChemMixFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class ChemMixFactory : public TaskFactoryBase {

  public:

    ChemMixFactory( const ApplicationCommon* arches );
    ~ChemMixFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset( const std::string subset ) {

      if ( subset == _all_tasks_str ){

        return _active_tasks;

      } else {

        throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset in CheMixFactory, which means there is no specific implementation for this factory.",__FILE__,__LINE__);

      }
    }

    void add_task( ProblemSpecP& db );

    void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart );

    /** @brief Helper to apply BCs to all table evaluators for this factory **/
    void schedule_applyBCs( const LevelP& level,
                            SchedulerP& sched,
                            const MaterialSet* matls,
                            const int time_substep );

  protected:

  private:

    std::vector<std::string> m_task_order;

  };
}
#endif

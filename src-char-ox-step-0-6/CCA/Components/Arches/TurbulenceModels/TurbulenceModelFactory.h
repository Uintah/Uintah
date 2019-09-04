#ifndef UT_TurbulenceModelFactory_h
#define UT_TurbulenceModelFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class TurbulenceModelFactory : public TaskFactoryBase {

  public:

    TurbulenceModelFactory( );
    ~TurbulenceModelFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset( const std::string subset ) {

      if ( subset == "momentum_closure" ){
        return m_momentum_closure_tasks;
      } else if ( subset == "all_tasks" ){
        return _active_tasks;
      } else {
        throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset in TurbulenceModelFactory, which means there is no specific implementation for this factory.",__FILE__,__LINE__);
      }


    }

    void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart );

  protected:

  private:

    std::vector<std::string> m_momentum_closure_tasks;

  };
}
#endif

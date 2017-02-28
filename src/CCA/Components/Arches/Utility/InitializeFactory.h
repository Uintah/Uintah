#ifndef UT_InitializeFactory_h
#define UT_InitializeFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class InitializeFactory : public TaskFactoryBase {

  public:

    InitializeFactory();
    ~InitializeFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset(const std::string subset){

      if ( subset == _all_tasks_str ){
        return _active_tasks;
      }

      throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset, which means there is no implementation for this factory.",__FILE__,__LINE__);
      
    }

    void schedule_initialization( const LevelP& level,
                                  SchedulerP& sched,
                                  const MaterialSet* matls,
                                  bool doing_restart );


  protected:


  private:



  };
}
#endif

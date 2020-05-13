#ifndef SRC_CCA_COMPONENTS_ARCHES_ARCHESEXAMPLES_EXAMPLEFACTORY_H_
#define SRC_CCA_COMPONENTS_ARCHES_ARCHESEXAMPLES_EXAMPLEFACTORY_H_

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{
  namespace ArchesExamples{
  class ExampleFactory : public TaskFactoryBase {

  public:

	ExampleFactory( const ApplicationCommon* arches );
    ~ExampleFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset(const std::string subset){

      if ( subset == "poisson1" ){
        return _poisson1_tasks;
      }else if ( subset == _all_tasks_str ){
        return _active_tasks;
      } else {
        throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset in ExampleFactory, which means there is no specific implementation for this factory for: "+subset,__FILE__,__LINE__);
      }
    }

    void add_task( ProblemSpecP& db );
    void schedule_initialization( const LevelP& level,
                                                     SchedulerP& sched,
                                                     const MaterialSet* matls,
                                                     bool doing_restart );
  protected:

  private:

    std::vector<std::string> _poisson1_tasks;  ///group of Poisson1 tasks

  };
  };
}


#endif /* SRC_CCA_COMPONENTS_ARCHES_ARCHESEXAMPLES_EXAMPLEFACTORY_H_ */

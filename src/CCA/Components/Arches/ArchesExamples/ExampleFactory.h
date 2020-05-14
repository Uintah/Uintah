#ifndef SRC_CCA_COMPONENTS_ARCHES_ARCHESEXAMPLES_EXAMPLEFACTORY_H_
#define SRC_CCA_COMPONENTS_ARCHES_ARCHESEXAMPLES_EXAMPLEFACTORY_H_

/*
 * Steps to create a new example using Arches
 * 1. Create a .h and .cc files based on ArchesExamples/Poisson1.h and .cc. (Check Poisson1.h for detailed comments)
 * 2. Add a new vector for group of new tasks (e.g. _poisson1_tasks)
 * 3. Add a new "if" condition in retrieve_task_subset for the new task e.g.:
 *     if ( subset == "poisson1" ){
 *       return _poisson1_tasks;
 *     }
 * 4. Update register_all_tasks to create and register the new arches task. e.g.:
 *     if(type=="poisson1"){
 *       TaskInterface::TaskBuilder* tsk = scinew Poisson1::Builder( type, 0 );
 *       _poisson1_tasks.push_back(type);
 *       register_task( type, tsk, db );
 *     }
 */

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{
  namespace ArchesExamples{
  class ExampleFactory : public TaskFactoryBase {

  public:

    ExampleFactory( const ApplicationCommon* arches );
    ~ExampleFactory();

    void register_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset(const std::string subset){

      if ( subset == "poisson1" ){ //3. Add a new "if" condition in retrieve_task_subset for the new task
        return _poisson1_tasks;
      }else if ( subset == _all_tasks_str ){
        return _active_tasks;
      } else {
        throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset in ExampleFactory, which means there is no specific implementation for this factory for: "+subset,__FILE__,__LINE__);
      }
    }

  protected:

  private:

    std::vector<std::string> _poisson1_tasks;  // group of Poisson1 tasks. (2. Add a new vector for groupd of new tasks)

  };
  };
}


#endif /* SRC_CCA_COMPONENTS_ARCHES_ARCHESEXAMPLES_EXAMPLEFACTORY_H_ */

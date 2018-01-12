#ifndef UT_UtilityFactory_h
#define UT_UtilityFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class UtilityFactory : public TaskFactoryBase {

  public:

    UtilityFactory();
    ~UtilityFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset(const std::string subset){

      if ( subset == _all_tasks_str ){
        return _active_tasks;
      }
      
      throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset, which means there is no implementation for this factory.",__FILE__,__LINE__);
    }

    void add_task( ProblemSpecP& db );

  protected:


  private:



  };
}
#endif

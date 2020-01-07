#ifndef UT_UtilityFactory_h
#define UT_UtilityFactory_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <string>

namespace Uintah{

  class UtilityFactory : public TaskFactoryBase {

  public:

    UtilityFactory( const ApplicationCommon* arches );
    ~UtilityFactory();

    void register_all_tasks( ProblemSpecP& db );

    void build_all_tasks( ProblemSpecP& db );

    std::vector<std::string> retrieve_task_subset(const std::string subset){

      if ( subset == "base_tasks" ){
        return _base_tasks;
      }
      else if ( subset == "mass_flow_rate" ){
        return _mass_flow_rate;
      }
      else if ( subset == _all_tasks_str ){
        return _active_tasks;
      }
      else {
        throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset, which means there is no implementation for this factory.",__FILE__,__LINE__);
      }
    }

    void add_task( ProblemSpecP& db );

  protected:

  private:

    std::vector<std::string> _base_tasks;
    std::vector<std::string> _mass_flow_rate;

  };
}
#endif

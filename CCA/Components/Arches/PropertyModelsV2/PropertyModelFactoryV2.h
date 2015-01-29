#ifndef UT_PropertyModelFactoryV2_h
#define UT_PropertyModelFactoryV2_h

#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <Core/Grid/SimulationState.h>
#include <string>

namespace Uintah{ 

  class PropertyModelFactoryV2 : public TaskFactoryBase { 

  public: 

    PropertyModelFactoryV2( SimulationStateP& shared_state ); 
    ~PropertyModelFactoryV2(); 

    void register_all_tasks( ProblemSpecP& db ); 

    void build_all_tasks( ProblemSpecP& db ); 

    std::vector<std::string> retrieve_task_subset( const std::string subset ) { 

      throw InvalidValue("Error: Accessing the base class implementation of retrieve_task_subset in PropertyModelFactoryV2, which means there is no specific implementation for this factory.",__FILE__,__LINE__); 

    }

  protected: 

  private: 

    SimulationStateP _shared_state; 

  };
}
#endif 
